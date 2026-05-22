from app.config.settings import get_settings


class OpenRouterClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    @property
    def model(self):
        if self._model is None and self.settings.openrouter_api_key:
            try:
                from langchain_openrouter import ChatOpenRouter

                client_kwargs = {
                    "api_key": self.settings.openrouter_api_key,
                    "model": self.settings.openrouter_model,
                    "temperature": 0.0,
                }
                if self.settings.openrouter_base_url:
                    client_kwargs["base_url"] = self.settings.openrouter_base_url

                self._model = ChatOpenRouter(**client_kwargs)
            except Exception:
                self._model = False
        return self._model

    def generate(self, prompt: str) -> tuple[str | None, dict[str, int]]:
        model = self.model
        if not model:
            return None, {"prompt_tokens": len(prompt.split()), "completion_tokens": 0}
        try:
            response = model.invoke(prompt)
            text = getattr(response, "content", None)
        except Exception:
            return None, {"prompt_tokens": len(prompt.split()), "completion_tokens": 0}
        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": len((text or "").split())}
        return text, usage

    def stream(self, prompt: str):
        model = self.model
        if not model:
            return

        try:
            for chunk in model.stream(prompt):
                text = getattr(chunk, "content", None)
                if text:
                    yield text
        except Exception:
            return
