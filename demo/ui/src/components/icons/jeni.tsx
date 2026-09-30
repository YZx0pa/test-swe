// Jeni demo: a plain mark for the header, in place of the LangGraph logo.
export function JeniMark({
  width = 32,
  height = 32,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) {
  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <rect
        width="32"
        height="32"
        rx="8"
        fill="#3b5bdb"
      />
      <path
        d="M19.5 8.5v10.25a4.75 4.75 0 0 1-9.5 0"
        stroke="#fff"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}
