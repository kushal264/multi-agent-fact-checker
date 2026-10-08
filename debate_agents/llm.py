from abc import ABC, abstractmethod
import os
import re
import time

class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiLLM(BaseLLM):

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        api_key: str = None,
    ):
        from google import genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. "
                "Set it as an environment variable."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str):

        max_retries = 5

        for attempt in range(max_retries):

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text

            except Exception as e:

                error_text = str(e)

                # Retry temporary server/capacity errors
                if "503" in error_text or "UNAVAILABLE" in error_text:

                    if attempt == max_retries - 1:
                        raise

                    wait_time = 2 ** attempt

                    print(
                        f"[Gemini] Temporary 503. "
                        f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)

                else:
                    raise


class DummyLLM(BaseLLM):

    def generate(self, prompt: str) -> str:

        claim_match = re.search(
            r"CLAIM:\s*(.+)",
            prompt,
        )

        claim = (
            claim_match.group(1).strip()
            if claim_match
            else "the claim"
        )

        evidence_ids = re.findall(
            r"\[(E\d+)\]",
            prompt,
        )

        # Judge quality request
        if "LOGICAL CONSISTENCY" in prompt.upper():

            argument_match = re.search(
                r"ARGUMENT \((FOR|AGAINST)\):\s*(.+)",
                prompt,
                re.DOTALL,
            )

            if argument_match:
                argument = argument_match.group(2).strip()

                if len(argument) > 30:
                    return "0.8"

            return "0.5"

        stance = (
            "FOR"
            if "Argue FOR" in prompt
            else "AGAINST"
        )

        cited = evidence_ids[:2]

        citation_str = " ".join(
            f"[{eid}]"
            for eid in cited
        )

        if stance == "FOR":

            return (
                f"The available evidence supports the claim "
                f"that {claim}. "
                f"{citation_str}"
            )

        return (
            f"The available evidence contradicts or "
            f"raises important doubts about the claim "
            f"that {claim}. "
            f"{citation_str}"
        )