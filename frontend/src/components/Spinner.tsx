interface SpinnerProps {
  size?: "sm" | "lg";
  ariaLabel?: string;
}

export function Spinner({ size = "sm", ariaLabel = "Carregando" }: SpinnerProps) {
  return (
    <span
      className={`spinner${size === "lg" ? " spinner--lg" : ""}`}
      role="status"
      aria-label={ariaLabel}
    />
  );
}
