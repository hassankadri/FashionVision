import './PrivacyPolicy.css';

export default function PrivacyPolicy() {
  return (
    <section className="privacy-page page-width" id="privacy" aria-labelledby="privacy-heading">
      <a href="#top" className="privacy-back">← Back to FashionVision</a>

      <header className="privacy-heading">
        <span className="eyebrow">PRIVACY / FASHIONVISION</span>
        <h1 id="privacy-heading" tabIndex={-1}>Privacy Policy</h1>
        <p className="privacy-date">Effective date: October 12, 2026</p>
        <p className="privacy-lead">
          This policy explains how FashionVision handles image uploads and technical
          information. It describes current application behavior and distinguishes it
          from processing that depends on the hosting infrastructure.
        </p>
      </header>

      <div className="privacy-content">
        <section aria-labelledby="privacy-images">
          <h2 id="privacy-images">Image uploads and AI processing</h2>
          <p>
            When you request a prediction or explanation, your selected image is sent
            to the application's backend and processed in application/server memory.
            The application does not intentionally save uploaded images to permanent storage.
          </p>
          <p>
            MobileNetV3 and CLIP predictions, and Grad-CAM explanations, run through
            the application's backend. Uploaded images are not sent to an external
            AI API by the application. The Fashion-MNIST browser CNN runs locally
            in your browser.
          </p>
        </section>

        <section aria-labelledby="privacy-personal">
          <h2 id="privacy-personal">Personal information</h2>
          <p>
            The application currently has no user accounts or user database.
            Uploaded images and their embedded metadata may themselves contain personal
            information. Please avoid submitting unnecessary personal or sensitive information.
          </p>
          <p>
            Selected images, previews, and prediction results are held in browser memory
            while you use the interface. Clearing an image removes the application's
            references to it; it does not delete the original file from your device.
          </p>
        </section>

        <section aria-labelledby="privacy-logs">
          <h2 id="privacy-logs">Server and infrastructure logs</h2>
          <p>
            Servers and hosting infrastructure may record IP addresses, request paths,
            timestamps, response status codes, and diagnostic errors. The application
            does not deliberately log uploaded filenames, image contents, or successful
            prediction results in its web request handlers. Infrastructure logging
            and retention depend on the hosting configuration.
          </p>
        </section>

        <section aria-labelledby="privacy-tracking">
          <h2 id="privacy-tracking">Cookies and tracking</h2>
          <p>
            No application-level analytics, cookies, or browser storage tracking
            is currently used by FashionVision. In-memory interface state supports
            image previews and results rather than tracking. Hosting providers may
            introduce their own cookies or other technologies, depending on the deployment.
          </p>
        </section>

        <section aria-labelledby="privacy-third-parties">
          <h2 id="privacy-third-parties">Third-party services</h2>
          <p>
            Google Fonts is loaded externally for the website's typography. Your browser
            therefore contacts Google's font servers, exposing technical request
            information such as your IP address and browser information.
          </p>
          <p>
            Hosting providers may process technical information, uploaded image
            requests, and prediction or explanation responses as part of delivering
            the service. Their processing, storage locations, and retention practices
            depend on the selected providers and configuration.
          </p>
        </section>

        <section aria-labelledby="privacy-retention">
          <h2 id="privacy-retention">Data retention</h2>
          <p>
            The server-side application does not intentionally retain uploaded images after
            processing. In-memory processing is not a guarantee that every copy is
            immediately or permanently erased. Infrastructure buffering, logs, and
            retention depend on hosting configuration; no fixed infrastructure
            retention period is specified here.
          </p>
        </section>

        <section aria-labelledby="privacy-security">
          <h2 id="privacy-security">Security</h2>
          <p>
            The application limits upload sizes and supported image formats and uses
            image validation in its processing paths. These precautions reduce some
            risks but cannot guarantee absolute security. Transmission security,
            infrastructure access controls, and other hosting protections depend on
            the deployment. No method of transmission or processing is completely secure.
          </p>
        </section>

        <section aria-labelledby="privacy-contact">
          <h2 id="privacy-contact">Contact</h2>
          <p>
            For questions about privacy or this policy, contact Hassan Kadri at{' '}
            <a href="mailto:hassankadri.dev@gmail.com">hassankadri.dev@gmail.com</a>.
          </p>
          <p>
            If you contact us by email, your email address and message are provided
            to us and handled by the email service used for correspondence.
          </p>
        </section>
      </div>

      <div className="privacy-bottom">
        <a href="#top" className="privacy-back">Return to FashionVision →</a>
      </div>
    </section>
  );
}
