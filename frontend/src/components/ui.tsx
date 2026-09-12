/**
 * Shared interface primitives.
 *
 * Small and deliberately unabstracted: each is a thin wrapper carrying the design
 * tokens and the accessibility details that are easy to forget - touch target size,
 * focus handling, `aria-live` on things that change.
 */

import type { ButtonHTMLAttributes, ReactNode } from "react";
import { useEffect, useId, useRef, useState } from "react";
import type { Severity } from "../api/types";
import "./ui.css";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "md" | "lg";
  block?: boolean;
  busy?: boolean;
  icon?: ReactNode;
};

export function Button({
  variant = "secondary",
  size = "md",
  block = false,
  busy = false,
  icon,
  children,
  className = "",
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`btn btn--${variant} btn--${size} ${block ? "btn--block" : ""} ${className}`}
      disabled={disabled || busy}
      // Announce the busy state rather than only showing a spinner.
      aria-busy={busy || undefined}
      {...rest}
    >
      {busy ? <span className="btn__spinner" aria-hidden="true" /> : icon}
      <span>{children}</span>
    </button>
  );
}

export function Card({
  children,
  className = "",
  as: Tag = "section",
}: {
  children: ReactNode;
  className?: string;
  as?: "section" | "div" | "article" | "aside";
}) {
  return <Tag className={`card ${className}`}>{children}</Tag>;
}

const SEVERITY_LABEL: Record<Severity, string> = {
  ok: "Healthy condition",
  caution: "Attention recommended",
  alert: "Defect suspected",
  unknown: "Unable to assess",
};

/**
 * Status indicator.
 *
 * Colour is never the only signal: each severity also carries a distinct glyph and a
 * text label, so the meaning survives colour blindness and greyscale printing.
 */
export function StatusBadge({ severity, label }: { severity: Severity; label?: string }) {
  const glyph = { ok: "✓", caution: "!", alert: "▲", unknown: "?" }[severity];
  return (
    <span className={`badge badge--${severity}`}>
      <span className="badge__glyph" aria-hidden="true">
        {glyph}
      </span>
      {label ?? SEVERITY_LABEL[severity]}
    </span>
  );
}

/**
 * Compact status indicator for dense lists.
 *
 * A full badge is too heavy in a list row, but a bare coloured pill reads as
 * decoration rather than information - and colour alone fails for colour-blind users
 * and in greyscale. This pairs a dot with a short word, and always carries an
 * accessible name.
 */
const SEVERITY_SHORT: Record<Severity, string> = {
  ok: "Healthy",
  caution: "Check",
  alert: "Defect",
  unknown: "Unknown",
};

export function StatusDot({ severity }: { severity: Severity }) {
  return (
    <span className={`statusdot statusdot--${severity}`}>
      <span className="statusdot__dot" aria-hidden="true" />
      <span className="statusdot__text">{SEVERITY_SHORT[severity]}</span>
    </span>
  );
}

/** A progress meter that states its value in words as well as width. */
export function Meter({
  value,
  label,
  severity = "ok",
}: {
  value: number;
  label: string;
  severity?: Severity;
}) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div className="meter">
      <div className="meter__head">
        <span className="meter__label">{label}</span>
        <span className="meter__value">{pct}%</span>
      </div>
      <div
        className="meter__track"
        role="meter"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${pct} percent`}
      >
        <div className={`meter__fill meter__fill--${severity}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

/**
 * Collapsible section, used for everything technical.
 *
 * A native `<details>` would be simpler, but its open state is hard to animate
 * consistently and impossible to control from outside, and the technical sections need
 * both. The ARIA wiring reproduces what `<details>` gives for free.
 */
export function Disclosure({
  title,
  subtitle,
  children,
  defaultOpen = false,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <div className={`disclosure ${open ? "is-open" : ""}`}>
      <button
        type="button"
        className="disclosure__trigger"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="disclosure__titles">
          <span className="disclosure__title">{title}</span>
          {subtitle && <span className="disclosure__subtitle">{subtitle}</span>}
        </span>
        <span className="disclosure__chevron" aria-hidden="true">
          ▾
        </span>
      </button>
      <div id={id} className="disclosure__panel" hidden={!open}>
        <div className="disclosure__inner">{children}</div>
      </div>
    </div>
  );
}

/** A labelled row of a key and a value, used throughout the report. */
export function DataRow({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
}) {
  return (
    <div className="datarow">
      <dt className="datarow__label">
        {label}
        {hint && <span className="datarow__hint">{hint}</span>}
      </dt>
      <dd className="datarow__value">{value}</dd>
    </div>
  );
}

/** Announces transient status to screen readers without stealing focus. */
export function LiveRegion({ message }: { message: string }) {
  return (
    <p className="sr-only" role="status" aria-live="polite">
      {message}
    </p>
  );
}

/** Traps nothing, but restores focus - enough for the simple panels here. */
export function useReturnFocus(active: boolean) {
  const previous = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (active) {
      previous.current = document.activeElement as HTMLElement | null;
    } else {
      previous.current?.focus?.();
    }
  }, [active]);
}
