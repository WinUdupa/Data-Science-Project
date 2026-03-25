""" Class for sampling new programs. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time
import re
import sys

from llmfe import evaluator
from llmfe import buffer
from llmfe import config as config_lib
import requests
import json
import http.client
import os

from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_evolution import PromptEvolver

load_dotenv()


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            meta_data: dict,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._meta_data = meta_data
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config
        self.__class__._global_samples_nums = 1

    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums//5 and self.__class__._global_samples_nums >= self._max_sample_nums//5:
                break

            prompt = self._database.get_prompt()
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            for sample in samples:
                sample = "\n    import pandas as pd\n    import numpy as np\n" + sample
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.data_input,
                    prompt.data_output,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1



def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding
    descriptions and the function signature. Preserves indentation.
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False

    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break

    if find_def_declaration:
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        return code

    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        super().__init__(samples_per_prompt)

        self._batch_inference = batch_inference
        self._url  = "http://127.0.0.1:5000/completions"
        self._trim = trim

        # ── Prompt Evolver ──────────────────────────────────────
        # Control via environment variables so no code changes needed:
        #   EVOLVE_MODEL    — which model to use for meta-prompting
        #   EVOLVE_INTERVAL — how many iterations between evolutions
        #   LOG_PATH        — where to save the prompt evolution log
        self.evolver = PromptEvolver(
            api_model          = os.environ.get("EVOLVE_MODEL", "llama-3.3-70b-versatile"),
            evolution_interval = int(os.environ.get("EVOLVE_INTERVAL", "5")),
            top_k              = 3,
            bad_k              = 2,
            log_path           = os.environ.get("LOG_PATH", "./logs/prompt_evolution"),
        )

        self._instruction_prompt = self.evolver.get_prompt()


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]

                return all_samples
            except Exception:
                continue


    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []

        # ── Evolve instruction prompt if interval reached ────────
        self._instruction_prompt = self.evolver.maybe_evolve()
        full_prompt = '\n'.join([self._instruction_prompt, prompt])

        # ── Verify API key ───────────────────────────────────────
        api_key = os.environ.get('API_KEY')
        if not api_key:
            raise ValueError("API_KEY not set. Add it to your .env file.")

        # ── Auto-detect endpoint ─────────────────────────────────
        if "gpt" in config.api_model.lower():
            host, endpoint = "api.openai.com", "/v1/chat/completions"
        else:
            host, endpoint = "api.groq.com", "/openai/v1/chat/completions"

        for _ in range(self._samples_per_prompt):
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    conn    = http.client.HTTPSConnection(host)
                    payload = json.dumps({
                        "max_tokens": 512,
                        "model":      config.api_model,
                        "messages":   [{"role": "user", "content": full_prompt}]
                    })
                    headers = {
                        'Authorization': f"Bearer {api_key}",
                        'User-Agent':    'Apifox/1.0.0 (https://apifox.com)',
                        'Content-Type':  'application/json'
                    }
                    conn.request("POST", endpoint, payload, headers)
                    res  = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))

                    if 'error' in data:
                        error = data['error']
                        print(f"API error: {error}")
                        if error.get('code') == 'rate_limit_exceeded':
                            message = error.get('message', '')
                            match   = re.search(r'try again in ([0-9.]+)s', message)
                            wait    = float(match.group(1)) + 1 if match else 60
                            print(f"Rate limited. Waiting {wait:.1f}s...")
                            time.sleep(wait)
                        else:
                            time.sleep(2)
                        continue

                    response = data['choices'][0]['message']['content']

                    if self._trim:
                        response = _extract_body(response, config)

                    # Record in evolver (score unknown here, updated later)
                    self.evolver.record(score=None, feature_fn=response)

                    all_samples.append(response)
                    break

                except Exception as e:
                    print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                    time.sleep(2)
            else:
                print("Max retries reached. Skipping this sample.")
                all_samples.append("")

        return all_samples

    def update_last_score(self, score: float):
        """
        Hook this into the evaluator to give accurate scores to the evolver.
        Call after each feature is evaluated:
            llm.update_last_score(score=0.834)
        """
        if self.evolver._records:
            self.evolver._records[-1].score = score


    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1

        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }

        headers  = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            response = response.json()["content"]
            return response if self._batch_inference else response[0]