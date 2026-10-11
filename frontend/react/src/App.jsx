
import { useEffect, useRef, useState } from 'react';
import './crop.css';

const MAX_BYTES = 15 * 1024 * 1024;
const ALLOWED = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp']);
const clamp = number => Math.min(1, Math.max(0, Number(number) || 0));
const pct = number => `${(clamp(number) * 100).toFixed(1)}%`;
const filename = name => name.length > 30 ? `${name.slice(0, 26)}…` : name;

async function postImage(endpoint, file, signal) {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': file.type },
    body: file,
    signal,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (HTTP ${response.status}).`);
  }
  return data;
}

function Arrow({ diagonal = false }) {
  return <span aria-hidden="true">{diagonal ? '↗' : '→'}</span>;
}

function SectionNext({ to, label }) {
  return (
    <div className="section-jump">
      <a href={`#${to}`} aria-label={label} title={label} className="section-jump-link">
        <span>{label}</span>
        <span className="section-jump-circle" aria-hidden="true">↓</span>
      </a>
    </div>
  );
}

function Header({ connected }) {
  return (
    <header className="site-header">
      <div className="header-inner">
        <a href="#top" className="brand" aria-label="FashionVision home">
          <span className="brand-mark"><span>F</span><span>V</span></span>
          <span className="brand-type">Fashion<span>Vision</span><small>AI / LAB</small></span>
        </a>
        <nav className="top-nav" aria-label="Main navigation">
          <a href="#studio">Studio</a>
          <a href="#gallery">Test gallery</a>
          <a href="#about">About the models</a>
        </nav>
        <div className="connection" title="Health reports CLIP readiness; MobileNet availability is checked when used">
          <span className={connected ? 'connection-dot online' : 'connection-dot'} />
          <span>{connected ? 'API connected' : 'API offline'}</span>
        </div>
      </div>
    </header>
  );
}

function ShirtArtwork() {
  return (
    <div className="hero-art" aria-hidden="true">
      <div className="art-topline"><span>VISION SYSTEM / 01</span><span><i /> MODEL ONLINE</span></div>
      <div className="art-grid" />
      <div className="orbital orbital-outer" />
      <div className="orbital orbital-inner" />
      <svg className="shirt-drawing" viewBox="0 0 400 450" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M135 95 L168 78 L234 78 L265 95 L340 150 L305 202 L271 183 L272 369 L130 369 L129 183 L95 202 L60 150 Z" fill="#D4D6D6" stroke="#FAFAFA" strokeWidth="2"/>
        <path d="M168 78 C175 116 226 116 234 78" stroke="#56595B" strokeWidth="14" />
        <path d="M135 95 L167 79 M265 95 L234 79" stroke="#8C9091" strokeWidth="3"/>
        <path d="M132 184 L130 370 M271 185 L272 369" stroke="#A5A6A7" strokeWidth="2"/>
      </svg>
      <div className="scan-line" />
      <span className="art-cross cross-one">+</span><span className="art-cross cross-two">+</span>
      <div className="art-label"><i /> IMAGE CLASSIFICATION <strong>10 CLASSES</strong></div>
      <div className="art-bottomline"><span>TRANSFER LEARNING</span><span>MOBILENETV3</span></div>
    </div>
  );
}

function Hero() {
  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="page-width hero-inner">
        <div className="hero-copy">
          <div className="hero-eyebrow"><span /> COMPUTER VISION / INTERACTIVE DEMO</div>
          <h1 id="hero-title">A new perspective <em>on fashion.</em></h1>
          <p>Upload a product photo, explore the model's predictions, and see which image regions influenced its decision.</p>
          <div className="hero-actions">
            <a className="hero-primary" href="#studio">Open vision studio <Arrow diagonal /></a>
            <a className="hero-secondary" href="#gallery">Explore the test gallery <Arrow /></a>
          </div>
          <div className="hero-metrics">
            <div><strong>99.25<span>%</span></strong><small>Product-image test accuracy</small></div>
            <div><strong>10</strong><small>Product categories</small></div>
            <div><strong>3</strong><small>Model experiences</small></div>
          </div>
        </div>
        <ShirtArtwork />
      </div>
      <SectionNext to="studio" label="EXPLORE THE STUDIO" />
    </section>
  );
}

function Ranking({ items, max = 3 }) {
  const ranked = (Array.isArray(items) ? items : [])
    .filter(item => item && typeof item.label === 'string' && Number.isFinite(Number(item.probability)))
    .slice(0, max);
  return <div className="ranking">
    {ranked.map((item, index) => <div className="rank" key={`${item.label}-${index}`}>
      <div className="rank-label"><span className="rank-index">{String(index + 1).padStart(2, '0')}</span><span>{item.label}</span><b>{pct(item.probability)}</b></div>
      <div className="rank-track"><div className="rank-fill" style={{ width: pct(item.probability) }} /></div>
    </div>)}
    {!ranked.length && <p className="muted">No ranking data was returned.</p>}
  </div>;
}

// Cropping happens in the browser; the original file remains available.
function CropDialog({ originalFile, onCancel, onApply }) {
  const [url, setUrl] = useState('');
  const [natural, setNatural] = useState({ width: 0, height: 0 });
  const [display, setDisplay] = useState({ width: 0, height: 0 });
  const [selection, setSelection] = useState(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const imageRef = useRef(null);
  const pointerRef = useRef(null);
  const closeRef = useRef(null);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(originalFile);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [originalFile]);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeRef.current?.focus();
    const onKey = event => {
      if (event.key === 'Escape' && !saving) onCancel();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = previous;
      window.removeEventListener('keydown', onKey);
    };
  }, [onCancel, saving]);

  useEffect(() => {
    if (!natural.width || !natural.height) return;
    const measure = () => {
      const availableWidth = Math.max(160, Math.min(900, window.innerWidth - 112));
      const availableHeight = Math.max(120, Math.min(590, window.innerHeight * 0.57));
      const scale = Math.min(
        1,
        availableWidth / natural.width,
        availableHeight / natural.height
      );
      setDisplay({
        width: Math.round(natural.width * scale),
        height: Math.round(natural.height * scale),
      });
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [natural]);

  const position = event => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: clamp((event.clientX - bounds.left) / bounds.width),
      y: clamp((event.clientY - bounds.top) / bounds.height),
    };
  };

  const startSelection = event => {
    if (!natural.width || (event.pointerType === 'mouse' && event.button !== 0)) return;
    event.preventDefault();
    const point = position(event);
    pointerRef.current = { ...point, previous: selection };
    event.currentTarget.setPointerCapture(event.pointerId);
    setSelection({ x: point.x, y: point.y, width: 0, height: 0 });
    setMessage('');
  };

  const moveSelection = event => {
    if (!pointerRef.current) return;
    const point = position(event);
    const start = pointerRef.current;
    setSelection({
      x: Math.min(point.x, start.x),
      y: Math.min(point.y, start.y),
      width: Math.abs(point.x - start.x),
      height: Math.abs(point.y - start.y),
    });
  };

  const finishSelection = event => {
    if (!pointerRef.current) return;
    const previous = pointerRef.current.previous;
    pointerRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setSelection(current =>
      current && current.width > 0.025 && current.height > 0.025
        ? current
        : previous
    );
  };

  const applyCrop = async () => {
    const img = imageRef.current;
    if (!img || !img.complete || !natural.width || !selection || saving) return;
    setSaving(true);
    setMessage('');

    try {
      const sx = Math.floor(selection.x * natural.width);
      const sy = Math.floor(selection.y * natural.height);
      const sw = Math.min(natural.width - sx, Math.max(1, Math.round(selection.width * natural.width)));
      const sh = Math.min(natural.height - sy, Math.max(1, Math.round(selection.height * natural.height)));

      if (sw < 16 || sh < 16) {
        throw new Error('Select a slightly larger area.');
      }

      const scale = Math.min(1, 2400 / Math.max(sw, sh));
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(sw * scale));
      canvas.height = Math.max(1, Math.round(sh * scale));

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('Image cropping is not supported in this browser.');
      }

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve, reject) => {
        canvas.toBlob(
          value => value
            ? resolve(value)
            : reject(new Error('Could not save cropped image.')),
          'image/jpeg',
          0.94
        );
      });

      if (blob.size > MAX_BYTES) {
        throw new Error('The crop is too large. Choose a smaller region.');
      }

      const name = originalFile.name.replace(/\.[^.]+$/, '') || 'image';
      onApply(new File([blob], `${name}-crop.jpg`, { type: 'image/jpeg' }));
    } catch (err) {
      setMessage(err.message || 'Cropping failed.');
    } finally {
      setSaving(false);
    }
  };

  const validSelection = Boolean(
    selection && selection.width > 0.025 && selection.height > 0.025
  );

  const selectedPixels = validSelection
    ? `${Math.round(selection.width * natural.width)} × ${Math.round(selection.height * natural.height)} px`
    : 'Drag on the photograph to choose an area';

  return (
    <div className="crop-backdrop" onPointerDown={e => {
      if (e.target === e.currentTarget && !saving) onCancel();
    }}>
      <section
        className="crop-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="crop-heading"
      >
        <div className="crop-heading">
          <div>
            <span className="crop-kicker">IMAGE EDITOR / OPTIONAL</span>
            <h2 id="crop-heading">Crop the photograph.</h2>
            <p>Drag a rectangle around the item you want to classify.</p>
          </div>
          <button
            type="button"
            className="crop-close"
            ref={closeRef}
            onClick={onCancel}
            aria-label="Close crop editor"
          >
            ×
          </button>
        </div>

        <div className="crop-workspace">
          <div
            className="crop-photo-shell"
            style={display.width ? {
              width: display.width,
              height: display.height,
            } : undefined}
            onPointerDown={startSelection}
            onPointerMove={moveSelection}
            onPointerUp={finishSelection}
            onPointerCancel={finishSelection}
          >
            {url && (
              <img
                ref={imageRef}
                src={url}
                alt="Original photograph for crop selection"
                draggable={false}
                onLoad={e => setNatural({
                  width: e.currentTarget.naturalWidth,
                  height: e.currentTarget.naturalHeight,
                })}
              />
            )}
            {selection && (
              <div className="crop-selection" style={{
                left: `${selection.x * 100}%`,
                top: `${selection.y * 100}%`,
                width: `${selection.width * 100}%`,
                height: `${selection.height * 100}%`,
              }} />
            )}
          </div>
        </div>

        <div className="crop-actions">
          <span className="crop-dimensions" aria-live="polite">
            {selectedPixels}
          </span>

          <div className="crop-action-buttons">
            <button
              type="button"
              className="crop-secondary"
              onClick={() => setSelection(null)}
              disabled={!selection || saving}
            >
              Clear selection
            </button>
            <button
              type="button"
              className="crop-secondary"
              onClick={onCancel}
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="button"
              className="crop-confirm"
              onClick={applyCrop}
              disabled={!validSelection || saving}
            >
              {saving ? 'Preparing image…' : 'Apply crop'}
              <span aria-hidden="true">↗</span>
            </button>
          </div>
        </div>

        {message && <p className="crop-error" role="alert">{message}</p>}
        <p className="crop-footnote">
          Cropping runs in your browser. Only the selected image is
          sent when you analyze it; the original remains available.
        </p>
      </section>
    </div>
  );
}

function Studio() {
  const [mode, setMode] = useState('mobilenet');
  const [file, setFile] = useState(null);
  const [originalFile, setOriginalFile] = useState(null);
  const [cropOpen, setCropOpen] = useState(false);
  const [preview, setPreview] = useState('');
  const [busy, setBusy] = useState('');
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [imageView, setImageView] = useState('original');

  const inputRef = useRef(null);
  const controllerRef = useRef(null);
  const serialRef = useRef(0);

  const invalidate = () => {
    serialRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    setBusy('');
    setResult(null);
    setExplanation(null);
    setImageView('original');
    setError('');
  };

  useEffect(() => {
    if (!file) {
      setPreview('');
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const chooseFile = candidate => {
    if (!candidate) return;

    invalidate();
    setFile(null);
    setOriginalFile(null);

    if (!ALLOWED.has(candidate.type)) {
      setError('Choose a JPG, PNG, WebP, or BMP image.');
      return;
    }
    if (!candidate.size) {
      setError('The selected image is empty.');
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setError('Images must be 15 MB or smaller.');
      return;
    }

    setOriginalFile(candidate);
    setFile(candidate);
  };

  const applyCrop = cropped => {
    invalidate();
    setFile(cropped);
    setCropOpen(false);
  };

  const restoreOriginal = () => {
    if (!originalFile) return;
    invalidate();
    setFile(originalFile);
  };

  const changeMode = next => {
    if (next === mode) return;
    invalidate();
    setMode(next);
  };

  const analyze = async () => {
    if (!file || busy) return;

    invalidate();
    const id = serialRef.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy('predict');

    try {
      const data = await postImage(
        mode === 'mobilenet'
          ? '/api/predict/mobilenet'
          : '/api/predict',
        file,
        controller.signal
      );

      if (serialRef.current !== id) return;

      const ranks = mode === 'mobilenet' ? data.top3 : data.predictions;

      if (
        typeof data.prediction !== 'string' ||
        !Array.isArray(ranks) ||
        !ranks.length
      ) {
        throw new Error('The API returned an unexpected prediction format.');
      }

      setResult(data);
    } catch (e) {
      if (serialRef.current === id && e.name !== 'AbortError') {
        setError(e.message || 'Prediction failed.');
      }
    } finally {
      if (serialRef.current === id) {
        setBusy('');
        controllerRef.current = null;
      }
    }
  };

  const explain = async () => {
    if (!file || !result || busy || mode !== 'mobilenet') return;

    const id = serialRef.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy('explain');
    setError('');

    try {
      const data = await postImage(
        '/api/explain/mobilenet',
        file,
        controller.signal
      );

      if (serialRef.current !== id) return;

      if (
        data.prediction !== result.prediction ||
        data.explained_class !== result.prediction
      ) {
        throw new Error('The explanation did not match the current prediction.');
      }

      const png = data.heatmap_overlay_base64;
      if (
        typeof png !== 'string' ||
        !/^iVBORw0KGgo[A-Za-z0-9+/]*={0,2}$/.test(png)
      ) {
        throw new Error('The API did not return a valid PNG heatmap.');
      }

      setExplanation(data);
      setImageView('heatmap');
    } catch (e) {
      if (serialRef.current === id && e.name !== 'AbortError') {
        setError(e.message || 'Explanation failed.');
      }
    } finally {
      if (serialRef.current === id) {
        setBusy('');
        controllerRef.current = null;
      }
    }
  };

  const clear = () => {
    invalidate();
    setFile(null);
    setOriginalFile(null);
    setCropOpen(false);
    if (inputRef.current) inputRef.current.value = '';
  };

  const ranks = mode === 'mobilenet'
    ? result?.top3
    : result?.predictions;

  const overlay = explanation
    ? `data:image/png;base64,${explanation.heatmap_overlay_base64}`
    : '';

  return (
    <section
      className="studio section-frame"
      id="studio"
      aria-labelledby="studio-heading"
    >
      <div className="section-top">
        <div>
          <div className="eyebrow">01 / THE STUDIO</div>
          <h2 id="studio-heading">Analyze an image.</h2>
        </div>
        <p>Select a model. Add an image. Inspect the output.</p>
      </div>

      <div className="studio-grid">
        <div className="image-stage">
          <div className="stage-head">
            <span>INPUT / {file ? filename(file.name) : 'NO IMAGE SELECTED'}</span>
            <span>{file ? 'READY FOR ANALYSIS' : 'AWAITING IMAGE'}</span>
          </div>

          <div
            className={`stage-body ${dragging ? 'dragging' : ''}`}
            onDragOver={e => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={e => {
              e.preventDefault();
              setDragging(false);
            }}
            onDrop={e => {
              e.preventDefault();
              setDragging(false);
              chooseFile(e.dataTransfer.files?.[0]);
            }}
          >
            {file && preview ? (
              <div
                className="visual-frame"
                aria-label={
                  imageView === 'heatmap'
                    ? 'Image with Grad-CAM color overlay'
                    : 'Uploaded product image'
                }
              >
                <img
                  className="stage-image"
                  src={preview}
                  alt="Selected product"
                />
                {imageView === 'heatmap' && overlay && (
                  <img
                    className="heatmap-color-layer"
                    src={overlay}
                    alt=""
                    aria-hidden="true"
                  />
                )}
              </div>
            ) : (
              <div className="upload-empty">
                <span className="upload-outline">+</span>
                <h3>Drop your image here.</h3>
                <p>Or choose a file from your device.</p>
                <button
                  type="button"
                  className="quiet-button"
                  onClick={() => inputRef.current?.click()}
                >
                  Browse files <Arrow diagonal />
                </button>
                <small>JPEG, PNG, WEBP, BMP · MAX 15 MB</small>
              </div>
            )}

            <input
              type="file"
              ref={inputRef}
              accept="image/jpeg,image/png,image/webp,image/bmp"
              className="sr-only"
              onChange={e => {
                chooseFile(e.target.files?.[0]);
                e.target.value = '';
              }}
            />
          </div>

          <div className="stage-foot">
            <span>
              {file
                ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type.replace('image/', '').toUpperCase()}`
                : 'SUPPORTED / SINGLE PRODUCT IMAGE'}
            </span>

            <div className="stage-actions">
              {explanation && (
                <div
                  className="view-switch"
                  role="group"
                  aria-label="Display image view"
                >
                  <button
                    className={imageView === 'original' ? 'selected' : ''}
                    onClick={() => setImageView('original')}
                  >
                    Original
                  </button>
                  <button
                    className={imageView === 'heatmap' ? 'selected' : ''}
                    onClick={() => setImageView('heatmap')}
                  >
                    Grad-CAM
                  </button>
                </div>
              )}

              {file && (
                <>
                  <button
                    type="button"
                    className="text-button crop-trigger"
                    onClick={() => setCropOpen(true)}
                  >
                    Crop image
                  </button>

                  {originalFile && file !== originalFile && (
                    <button
                      type="button"
                      className="text-button"
                      onClick={restoreOriginal}
                    >
                      Restore original
                    </button>
                  )}

                  <button
                    type="button"
                    className="text-button"
                    onClick={() => inputRef.current?.click()}
                  >
                    Replace
                  </button>

                  <button
                    type="button"
                    className="text-button"
                    onClick={clear}
                  >
                    Remove
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        <aside className="analysis-panel">
          <div className="panel-overline">
            CONFIGURATION <span>LIVE MODEL</span>
          </div>

          <div
            className="model-switch"
            role="group"
            aria-label="Choose classification model"
          >
            <button
              type="button"
              className={mode === 'mobilenet' ? 'active' : ''}
              onClick={() => changeMode('mobilenet')}
              aria-pressed={mode === 'mobilenet'}
            >
              <strong>MobileNetV3</strong>
              <small>Product classifier</small>
            </button>

            <button
              type="button"
              className={mode === 'clip' ? 'active' : ''}
              onClick={() => changeMode('clip')}
              aria-pressed={mode === 'clip'}
            >
              <strong>CLIP</strong>
              <small>Zero-shot comparison</small>
            </button>
          </div>

          <button
            type="button"
            className="run-button"
            onClick={analyze}
            disabled={!file || !!busy}
          >
            {busy === 'predict' ? 'Analyzing…' : 'Run analysis'}
            <Arrow diagonal />
          </button>

          <div className="panel-rule" />
          {error && (
            <p className="error-message" role="alert">{error}</p>
          )}

          {result ? (
            <div className="result" aria-live="polite">
              <div className="panel-overline">
                PREDICTION
                <span>{mode === 'mobilenet' ? 'TOP 3' : 'TOP 5'}</span>
              </div>

              <h3>{result.prediction}</h3>

              <Ranking
                items={ranks}
                max={mode === 'mobilenet' ? 3 : 5}
              />

              <p className="result-disclaimer">
                {mode === 'mobilenet'
                  ? 'These are uncalibrated softmax scores, not guaranteed confidence.'
                  : 'CLIP scores are relative similarity shares, not calibrated probabilities.'}
              </p>

              {mode === 'mobilenet' && (
                <div className="explain-area">
                  <div className="explain-title">
                    <strong>Visual explanation</strong>
                    <small>Class-specific Grad-CAM attribution</small>
                  </div>

                  <button
                    onClick={explain}
                    className="outline-button"
                    disabled={!!busy}
                  >
                    {busy === 'explain'
                      ? 'Computing gradients…'
                      : explanation
                        ? 'Regenerate heatmap'
                        : 'Generate heatmap'}
                    <Arrow diagonal />
                  </button>

                  {explanation && (
                    <p className="explain-note">
                      5 × 5 attribution resized to 160 × 160 ·{' '}
                      {explanation.attribution_status}.
                      Not segmentation or proof of causality.
                    </p>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="result-blank" aria-live="polite">
              <span className="result-big-number">—</span>
              <h3>No prediction yet.</h3>
              <p>
                Once you analyze an image, its predicted label
                and alternatives will appear here.
              </p>
            </div>
          )}
        </aside>
      </div>

      <p className="workspace-note">
        The model analyzes the selected image. Cropping can help isolate an item;
        independent natural-photo accuracy has not been measured.
        Grad-CAM remains a coarse 5 × 5 attribution.
      </p>

      <SectionNext to="gallery" label="EXPLORE THE TEST GALLERY" />

      {cropOpen && originalFile && (
        <CropDialog
          originalFile={originalFile}
          onApply={applyCrop}
          onCancel={() => setCropOpen(false)}
        />
      )}
    </section>
  );
}

function samplePixels(base64) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 28;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(image, 0, 0, 28, 28);
        const rgba = ctx.getImageData(0, 0, 28, 28).data;
        const pixels = new Float32Array(784);
        for (let i = 0; i < 784; i++) pixels[i] = rgba[i * 4] / 255;
        resolve(pixels);
      } catch (error) {
        reject(error);
      }
    };
    image.onerror = () => reject(new Error('Could not decode gallery image.'));
    image.src = `data:image/png;base64,${base64}`;
  });
}

function Gallery() {
  const [gallery, setGallery] = useState(null);
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState(null);
  const [sampleResult, setSampleResult] = useState(null);
  const [error, setError] = useState('');
  const weightsRef = useRef(null);
  const selectId = useRef(0);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch('/frontend/model_weights.json').then(r => {
        if (!r.ok) throw new Error('Cannot load CNN weights.');
        return r.json();
      }),
      fetch('/frontend/sample_images.json').then(r => {
        if (!r.ok) throw new Error('Cannot load sample images.');
        return r.json();
      }),
    ]).then(([weights, samples]) => {
      if (cancelled) return;
      if (!window.FashionCNN) {
        throw new Error('Browser CNN script was not loaded.');
      }
      weightsRef.current = window.FashionCNN.load(weights);
      setGallery({ samples, labels: weights.class_names });
    }).catch(e => {
      if (!cancelled) setError(e.message);
    });
    return () => { cancelled = true; };
  }, []);

  const chooseSample = async (sample, index) => {
    const id = ++selectId.current;
    setSelected(index);
    setSampleResult(null);
    setError('');

    try {
      const pixels = await samplePixels(sample.png);
      const scores = window.FashionCNN.predict(
        pixels,
        weightsRef.current
      );
      if (id !== selectId.current) return;

      const ranked = gallery.labels
        .map((label, classId) => ({
          label,
          probability: scores[classId],
          classId,
        }))
        .sort((a, b) => b.probability - a.probability);

      setSampleResult({
        ranked,
        correct: ranked[0].classId === sample.label,
        expected: gallery.labels[sample.label],
      });
    } catch (e) {
      if (id === selectId.current) {
        setError(e.message || 'Could not classify sample.');
      }
    }
  };

  const shown = gallery?.samples
    .map((sample, index) => ({ sample, index }))
    .filter(({ sample }) =>
      filter === 'all' || String(sample.label) === filter
    ) || [];

  return (
    <section className="gallery-section" id="gallery" aria-labelledby="gallery-heading">
      <div className="page-width">
        <div className="gallery-intro section-heading">
          <div>
            <div className="eyebrow">02 / BROWSER CNN</div>
            <h2 id="gallery-heading">The original test bench.</h2>
            <p>
              Forty Fashion-MNIST test samples, classified directly
              in your browser using the original five-view CNN.
            </p>
          </div>
          <span className="section-aside">28 × 28 PIXELS · NO API REQUIRED</span>
        </div>

        <div className="gallery-layout">
          <div className="gallery-controls gallery-card">
            <div className="gallery-toolbar">
              <div>
                <h3>Test image gallery</h3>
                <p>Choose a sample to classify it.</p>
              </div>

              <label className="gallery-filter">
                Category
                <select value={filter} onChange={e => {
                  setFilter(e.target.value);
                  selectId.current++;
                  setSelected(null);
                  setSampleResult(null);
                  setError('');
                }}>
                  <option value="all">All classes</option>
                  {gallery?.labels.map((name, index) => (
                    <option value={String(index)} key={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="gallery-tiles" aria-label="Fashion-MNIST test image gallery">
              {shown.map(({ sample, index }) => (
                <button
                  key={index}
                  type="button"
                  className={`sample-tile ${selected === index ? 'selected' : ''}`}
                  onClick={() => chooseSample(sample, index)}
                  title={sample.label_name}
                  aria-label={`Classify sample ${index + 1}, ${sample.label_name}`}
                >
                  <img
                    src={`data:image/png;base64,${sample.png}`}
                    alt=""
                    loading="lazy"
                  />
                </button>
              ))}
              {!gallery && !error && (
                <span className="muted">Loading saved samples…</span>
              )}
            </div>

            <div className="gallery-foot">
              {shown.length} samples shown · 5 original/shifted views per prediction
            </div>
          </div>

          <div className="gallery-output gallery-card" aria-live="polite">
            {sampleResult ? (
              <>
                <div className="eyebrow">BROWSER INFERENCE / 05 VIEWS</div>
                <div className="sample-summary">
                  <img
                    src={`data:image/png;base64,${gallery.samples[selected].png}`}
                    alt="Selected Fashion-MNIST sample"
                  />
                  <div>
                    <small>PREDICTED CLASS</small>
                    <h3>{sampleResult.ranked[0].label}</h3>
                    <span className="sample-verdict">
                      {sampleResult.correct
                        ? 'Correct prediction'
                        : `Expected: ${sampleResult.expected}`}
                    </span>
                  </div>
                </div>
                <div className="gallery-rank-title">TOP 3 OUTPUTS</div>
                <Ranking items={sampleResult.ranked} max={3} />
                <p className="gallery-caption">
                  This CNN runs in your browser on 28 × 28 grayscale images.
                  It is separate from the product-photo models.
                </p>
              </>
            ) : (
              <div className="gallery-empty">
                <div className="pixel-mark">28<span>×</span>28</div>
                <h3>Choose a test image.</h3>
                <p>
                  The CNN runs locally in your browser.
                  No sample is sent to the server.
                </p>
              </div>
            )}
            {error && (
              <p className="error-message" role="alert">{error}</p>
            )}
          </div>
        </div>

        <SectionNext to="about" label="ABOUT THE MODELS" />
      </div>
    </section>
  );
}

function About() {
  return (
    <section className="about-section" id="about" aria-labelledby="about-heading">
      <div className="page-width">
        <div className="section-heading about-heading">
          <div>
            <div className="eyebrow">03 / ABOUT THE PROJECT</div>
            <h2 id="about-heading">Built, tested, and explained.</h2>
            <p>
              Three approaches to fashion recognition,
              each with a different purpose and evaluation.
            </p>
          </div>
          <span className="section-aside">TECHNICAL NOTES / FASHIONVISION</span>
        </div>

        <div className="model-cards">
          <article className="model-card">
            <span className="card-index">01 / PRODUCT CLASSIFIER</span>
            <h3>MobileNetV3Small</h3>
            <p>
              A pretrained image model fine-tuned on ten product categories.
              Uses 160 × 160 inputs and returns ranked class scores.
            </p>
            <div className="card-spec">
              <span>25,455</span><small>Deduplicated product images</small>
            </div>
            <div className="card-spec">
              <span>99.25%</span><small>Top-1 held-out accuracy</small>
            </div>
            <div className="card-footer">TRANSFER LEARNING · TENSORFLOW / KERAS</div>
          </article>

          <article className="model-card">
            <span className="card-index">02 / ZERO-SHOT MODEL</span>
            <h3>CLIP ViT-B/32</h3>
            <p>
              Compares an uploaded photo with text labels without
              training on this project's ten-category product dataset.
            </p>
            <div className="card-spec">
              <span>ONNX</span><small>Quantized local inference</small>
            </div>
            <div className="card-spec">
              <span>—</span><small>Independent photo accuracy not measured</small>
            </div>
            <div className="card-footer">IMAGE / TEXT SIMILARITY · ZERO SHOT</div>
          </article>

          <article className="model-card">
            <span className="card-index">03 / ORIGINAL EXPERIMENT</span>
            <h3>Fashion-MNIST CNN</h3>
            <p>
              An earlier convolutional network built for 28 × 28 grayscale
              clothing images. Try it on the forty saved test samples.
            </p>
            <div className="card-spec">
              <span>28 × 28</span><small>Grayscale input</small>
            </div>
            <div className="card-spec">
              <span>5 views</span><small>Original + four circular shifts</small>
            </div>
            <div className="card-footer">LOCAL BROWSER INFERENCE · JAVASCRIPT</div>
          </article>
        </div>

        <div className="research-detail">
          <div className="research-intro">
            <span className="eyebrow">EVALUATION METHOD</span>
            <h3>What the benchmark actually means.</h3>
          </div>
          <div className="research-text">
            <p>
              For the product classifier, the dataset was deduplicated and
              divided into 20,364 training, 2,545 validation, and 2,546
              test images using inferred product-family grouping.
              The held-out test contained 2,527 correct top-1 predictions,
              with 99.92% top-3 accuracy.
            </p>
            <p>
              The family grouping is inferred rather than guaranteed,
              so related images may still appear across splits.
              These numbers describe the held-out product-image dataset—not
              general performance on independent photographs.
            </p>
          </div>
        </div>

        <div className="research-detail research-last">
          <div className="research-intro">
            <span className="eyebrow">VISUAL EXPLANATIONS</span>
            <h3>Where the network looked.</h3>
          </div>
          <div className="research-text">
            <p>
              Grad-CAM uses gradients from the MobileNet backbone to generate
              a class-specific attribution map. The underlying map is 5 × 5
              and is enlarged to the model's 160 × 160 input view for inspection.
            </p>
            <p>
              It indicates regions associated with a prediction.
              It is not object segmentation, a guarantee of correctness,
              or proof that a particular feature caused the decision.
            </p>
          </div>
        </div>

        <footer className="page-footer">
          <div>
            <strong>FashionVision</strong>
            <span>Independent computer vision project</span>
          </div>
          <span>Built to make model outputs easier to inspect.</span>
          <a href="#top">Back to top ↑</a>
        </footer>
      </div>
    </section>
  );
}

export default function App() {
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let frame = 0;

    const cancelScroll = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    };

    const clickAnchor = event => {
      const anchor = event.target instanceof Element
        ? event.target.closest('a[href^="#"]')
        : null;

      if (
        !anchor ||
        event.defaultPrevented ||
        event.button !== 0 ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        event.altKey
      ) return;

      const id = anchor.getAttribute('href').slice(1);
      const target = document.getElementById(id);
      if (!target) return;

      event.preventDefault();
      cancelScroll();

      const offset = id === 'top'
        ? 0
        : window.innerWidth <= 580 ? 62 : 78;

      const end = Math.max(
        0,
        Math.min(
          document.documentElement.scrollHeight - window.innerHeight,
          target.getBoundingClientRect().top + window.scrollY - offset
        )
      );

      history.pushState(null, '', `#${id}`);

      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        window.scrollTo(0, end);
        return;
      }

      const start = window.scrollY;
      const distance = end - start;
      const duration = Math.min(
        1900,
        Math.max(1300, Math.abs(distance) * 0.75)
      );

      let started;
      const tick = time => {
        if (started === undefined) started = time;
        const t = Math.min(1, (time - started) / duration);
        const eased = t < 0.5
          ? 4 * t ** 3
          : 1 - (-2 * t + 2) ** 3 / 2;

        window.scrollTo(0, start + distance * eased);
        frame = t < 1 ? requestAnimationFrame(tick) : 0;
      };

      frame = requestAnimationFrame(tick);
    };

    document.addEventListener('click', clickAnchor);
    window.addEventListener('wheel', cancelScroll, { passive: true });
    window.addEventListener('touchstart', cancelScroll, { passive: true });

    return () => {
      cancelScroll();
      document.removeEventListener('click', clickAnchor);
      window.removeEventListener('wheel', cancelScroll);
      window.removeEventListener('touchstart', cancelScroll);
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/health', { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error();
        return response.json();
      })
      .then(data => setConnected(data?.status === 'ok'))
      .catch(() => setConnected(false));

    return () => controller.abort();
  }, []);

  return (
    <>
      <Header connected={connected} />
      <main>
        <Hero />
        <Studio />
        <Gallery />
        <About />
      </main>
    </>
  );
}
