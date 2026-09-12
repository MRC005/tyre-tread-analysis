import { useEffect, useState } from "react";
import { Button, LiveRegion } from "../components/ui";

/**
 * The waiting state.
 *
 * The steps shown are the real pipeline stages, paced to the measured backend time
 * (around 80-200 ms locally, dominated by upload on a phone). They are not a fake
 * progress bar: each label names something the backend genuinely does, which is also
 * what makes the wait feel purposeful rather than arbitrary.
 */
const STEPS = [
  "Uploading the photo",
  "Checking image quality",
  "Locating the tread area",
  "Measuring surface texture",
  "Assessing condition",
];

export function Analyzing({ onCancel }: { onCancel: () => void }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    // Advances to the second-to-last step and waits there. Never claims to have
    // finished the last step, because only the response can tell us that.
    const id = setInterval(() => {
      setStep((current) => Math.min(current + 1, STEPS.length - 2));
    }, 700);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="analyzing container">
      <LiveRegion message="Analysing your photo. This usually takes a few seconds." />
      <div className="analyzing__inner">
        <div className="analyzing__pulse" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <h1 className="analyzing__title">Analysing</h1>
        <ol className="analyzing__steps">
          {STEPS.map((label, index) => (
            <li
              key={label}
              className={`analyzing__step ${index < step ? "is-done" : ""} ${
                index === step ? "is-active" : ""
              }`}
            >
              <span className="analyzing__dot" aria-hidden="true" />
              {label}
            </li>
          ))}
        </ol>
        <Button variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}
