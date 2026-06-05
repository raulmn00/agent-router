import { useCallback, useState, type KeyboardEvent } from "react";

import { Spinner } from "./Spinner";

interface MessageInputProps {
  placeholder: string;
  submitLabel: string;
  busy: boolean;
  onSubmit: (text: string) => void;
  maxLength?: number;
}

export function MessageInput({
  placeholder,
  submitLabel,
  busy,
  onSubmit,
  maxLength = 2000,
}: MessageInputProps) {
  const [value, setValue] = useState("");
  const trimmed = value.trim();
  const canSubmit = trimmed.length > 0 && !busy;

  const submit = useCallback(() => {
    if (!canSubmit) return;
    onSubmit(trimmed);
  }, [canSubmit, trimmed, onSubmit]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="message-input">
      <textarea
        className="message-input__textarea"
        placeholder={placeholder}
        value={value}
        maxLength={maxLength}
        disabled={busy}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        aria-label="Mensagem"
      />
      <div className="message-input__row">
        <span className="message-input__hint">
          {value.length}/{maxLength} · Ctrl/⌘+Enter para enviar
        </span>
        <button
          type="button"
          className="message-input__btn"
          disabled={!canSubmit}
          onClick={submit}
        >
          {busy ? <Spinner /> : null}
          <span>{submitLabel}</span>
        </button>
      </div>
    </div>
  );
}
