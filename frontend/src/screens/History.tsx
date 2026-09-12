import { Button, Card, StatusBadge } from "../components/ui";
import type { HistoryEntry } from "../lib/history";

export function History({
  entries,
  onBack,
  onClear,
  onNew,
}: {
  entries: HistoryEntry[];
  onBack: () => void;
  onClear: () => void;
  onNew: () => void;
}) {
  return (
    <div className="history">
      <header className="screen-header container">
        <Button variant="ghost" onClick={onBack}>
          ← Back
        </Button>
        <h1 className="screen-header__title">This session</h1>
        <span className="screen-header__spacer" />
      </header>

      <div className="container stack">
        {entries.length === 0 ? (
          <Card className="empty">
            <p className="empty__title">No inspections yet</p>
            <p className="empty__text">Inspections you run will be listed here.</p>
          </Card>
        ) : (
          <>
            <ul className="history__list">
              {entries.map((entry) => (
                <li key={entry.id}>
                  <Card className="history__card">
                    {entry.thumbnail ? (
                      <img src={entry.thumbnail} alt="" className="history__thumb" />
                    ) : (
                      <div className="history__thumb history__thumb--empty" />
                    )}
                    <div className="history__body">
                      <div className="history__row">
                        <span className="history__label">{entry.label}</span>
                        <StatusBadge severity={entry.severity} label="" />
                      </div>
                      <p className="history__headline">{entry.headline}</p>
                      <p className="history__meta">
                        {entry.surface ?? "Surface not determined"} · Quality:{" "}
                        {entry.quality}
                        {entry.confidence !== null && (
                          <> · {Math.round(entry.confidence * 100)}% confidence</>
                        )}
                      </p>
                    </div>
                  </Card>
                </li>
              ))}
            </ul>
            <p className="muted-note">
              Kept on this device for this browsing session only. Nothing is uploaded or
              stored on a server, and it clears when you close the tab.
            </p>
          </>
        )}

        <div className="stack-sm">
          <Button variant="primary" size="lg" block onClick={onNew}>
            Inspect another tyre
          </Button>
          {entries.length > 0 && (
            <Button variant="ghost" block onClick={onClear}>
              Clear history
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
