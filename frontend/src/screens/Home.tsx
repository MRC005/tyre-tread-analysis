import { Button, Card, Disclosure, StatusDot } from "../components/ui";
import type { HistoryEntry } from "../lib/history";

/**
 * The entry screen.
 *
 * The framing is road safety, not machine learning: a driver wants to know whether a
 * tyre needs attention before a journey. What the tool cannot do is still stated
 * honestly, but it is no longer the first thing the page says. An earlier version led
 * with three "cannot" bullets, which made a working product read as a disclaimer.
 * Those facts now live one line down and behind "About this check", where someone who
 * wants them will find them and everyone else can get on with checking a tyre.
 */
export function Home({
  onStart,
  history,
  onOpenHistory,
  backendDown,
}: {
  onStart: () => void;
  history: HistoryEntry[];
  onOpenHistory: () => void;
  backendDown: boolean;
}) {
  return (
    <div className="home">
      <div className="container stack-lg">
        <header className="home__hero">
          <p className="home__eyebrow">Tyre safety screening</p>
          <h1 className="home__title">Check your tyres before you drive</h1>
          <p className="home__lede">
            Photograph a tyre and see whether its surface shows signs of wear, cracking
            or perished rubber — in a few seconds, with the reasoning shown.
          </p>
        </header>

        {backendDown && (
          <div className="notice notice--warn" role="status">
            <strong>Analysis service unreachable.</strong> You can still take a photo,
            but analysis will fail until the service is back.
          </div>
        )}

        <div className="home__cta">
          <Button variant="primary" size="lg" block onClick={onStart}>
            Check a tyre
          </Button>
          <p className="home__cta-note">
            Image-based screening. For tread-depth measurement or a definitive
            inspection, see a qualified tyre professional.
          </p>
        </div>

        {history.length > 0 && (
          <Card className="home__history">
            <div className="home__history-head">
              <h2 className="section-title">This session</h2>
              <Button variant="secondary" onClick={onOpenHistory}>
                View all ({history.length})
              </Button>
            </div>
            <ul className="home__history-list">
              {history.slice(0, 3).map((entry) => (
                <li key={entry.id} className="home__history-item">
                  {entry.thumbnail ? (
                    <img src={entry.thumbnail} alt="" className="home__history-thumb" />
                  ) : (
                    <div className="home__history-thumb home__history-thumb--empty" />
                  )}
                  <div className="home__history-text">
                    <span className="home__history-label">{entry.label}</span>
                    <span className="home__history-headline">{entry.headline}</span>
                  </div>
                  <StatusDot severity={entry.severity} />
                </li>
              ))}
            </ul>
          </Card>
        )}

        <section className="home__steps" aria-label="How it works">
          <h2 className="section-title">How it works</h2>
          <ol className="steps">
            <li className="steps__item">
              <span className="steps__num">1</span>
              <div>
                <strong>Photograph the tyre</strong>
                <p>
                  The camera guides your framing and tells you when the shot is good
                  enough.
                </p>
              </div>
            </li>
            <li className="steps__item">
              <span className="steps__num">2</span>
              <div>
                <strong>The photo is checked first</strong>
                <p>
                  Too dark, blurred or too far away, and you're asked to retake it
                  rather than given a guess.
                </p>
              </div>
            </li>
            <li className="steps__item">
              <span className="steps__num">3</span>
              <div>
                <strong>Read the result</strong>
                <p>
                  Condition, how confident the system is, and what to do about it.
                </p>
              </div>
            </li>
          </ol>
        </section>

        <section className="why" aria-label="Why this matters">
          <h2 className="section-title">Why this matters</h2>
          <p className="why__lede">
            Tread depth is what lets a tyre clear water and grip in an emergency stop.
            In India the legal minimum is <strong>1.6&nbsp;mm</strong> for cars and{" "}
            <strong>0.8&nbsp;mm</strong> for two- and three-wheelers, measured against
            the tread wear indicator moulded into the tyre — CMVR Rule 95.
          </p>
          <p className="why__lede">
            Checking that properly needs a gauge and a look at the tyre. Most of us
            don't do it between services. A photograph is something everyone can take,
            and visible wear, cracking and perished rubber are things a camera can
            genuinely pick up.
          </p>
          <p className="why__note">
            This tool is a prompt to look more carefully — not a measurement, and not a
            substitute for a gauge or a fitter.
          </p>
        </section>

        <Disclosure
          title="About this check"
          subtitle="What it can and can't tell you"
        >
          <div className="about stack">
            <div>
              <h3 className="about__title">What it looks for</h3>
              <p>
                Surface characteristics associated with worn tread, cracking and
                perished rubber — the things visible in a photograph of the tyre.
              </p>
            </div>
            <div>
              <h3 className="about__title">What it can't determine</h3>
              <ul className="about__list">
                <li>
                  Remaining tread depth. No photograph carries a scale reference, and
                  no public dataset provides measured depths to learn from.
                </li>
                <li>
                  Whether a tyre is legal or roadworthy — that depends on measured
                  depth against your local limit.
                </li>
                <li>
                  Internal damage, pressure, or anything not visible on the surface
                  photographed.
                </li>
              </ul>
            </div>
            <div>
              <h3 className="about__title">How it was built</h3>
              <p>
                Classical image processing — texture, edge and frequency analysis —
                with a calibrated classifier that abstains when it isn't confident.
                Every result shows the measurements behind it under Technical analysis.
              </p>
            </div>
            <p className="about__footnote">
              This is a screening aid, not a certified inspection. It supports a
              decision to get a tyre looked at; it does not replace one.
            </p>
          </div>
        </Disclosure>
      </div>
    </div>
  );
}
