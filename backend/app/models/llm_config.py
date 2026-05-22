import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config.settings import get_settings


class LLMConfigurationError(RuntimeError):
    pass


class LLMGenerationError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(self, prompt: str) -> tuple[str, dict[str, int]]:
        if not self.settings.openrouter_api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY is required to answer policy queries.")

        try:
            response = self._post_chat_completion(prompt)
            text = self._response_text(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMGenerationError(f"OpenRouter returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise LLMGenerationError(f"OpenRouter connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMGenerationError("OpenRouter timed out before returning a response.") from exc
        except Exception as exc:
            raise LLMGenerationError(f"OpenRouter did not return a response: {exc}") from exc

        if not text:
            raise LLMGenerationError("OpenRouter returned an empty response.")

        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split())}
        return text, usage

    def stream(self, prompt: str):
        text, _ = self.generate(prompt)
        yield text

    def _post_chat_completion(self, prompt: str) -> dict:
        body = {
            "model": self.settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        request = Request(
            self._chat_completions_url(),
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": self.settings.app_name,
            },
            method="POST",
        )
        with urlopen(request, timeout=self.settings.openrouter_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _chat_completions_url(self) -> str:
        base_url = self.settings.openrouter_base_url or "https://openrouter.ai/api/v1"
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    @staticmethod
    def _response_text(response: dict) -> str:
        choices = response.get("choices") or []
        if not choices:
            return ""

        first_choice = choices[0]
        message = first_choice.get("message") or {}
        content = message.get("content") or first_choice.get("text") or ""
        if isinstance(content, list):
            return "".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        return str(content or "").strip()
