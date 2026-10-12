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
            Small images are sent directly to the application's backend and processed
            in application/server memory. When private uploads are enabled, larger images
            are uploaded directly to private Vercel Blob storage and retrieved by the
            backend for predictions or explanations. The application uses this storage
            temporarily and does not offer public access to uploaded images.
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
            The application has no user accounts or user profiles. The private-upload
            feature stores temporary authorization records and security counters in a
            shared data store. Uploaded images and their embedded metadata may contain personal
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
            and retention depend on the hosting configuration. For private-upload abuse
            protection, the server uses the requester's IP address to create a keyed,
            pseudonymized rate-limit identifier. Temporary authorization records include
            generated object identifiers, declared image size and type, expiry, receipt
            hashes, and processing or cleanup state; they do not contain image bytes or
            prediction results.
          </p>
        </section>

        <section aria-labelledby="privacy-tracking">
          <h2 id="privacy-tracking">Cookies and tracking</h2>
          <p>
            FashionVision does not add application analytics or browser tracking cookies
            or storage. In-memory interface state supports image previews and results.
            Private-upload security counters are stored server-side for abuse protection,
            rather than advertising or behavioral analytics. Hosting providers may
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
            Vercel hosts the website and CLIP API and provides the private Blob store.
            When private uploads are enabled, Upstash Redis processes temporary upload
            authorization records and pseudonymized security counters. These services
            may also process technical request information. Their processing, storage
            locations, infrastructure logs, caches, and retention practices depend on
            the providers and deployment configuration.
          </p>
        </section>

        <section aria-labelledby="privacy-retention">
          <h2 id="privacy-retention">Data retention</h2>
          <p>
            For private uploads, the application attempts to delete the temporary object
            after prediction or explanation, including failures. A protected daily cleanup
            job is configured to remove abandoned objects older than one hour. This is a
            cleanup grace period, not a guaranteed deletion deadline: daily scheduling,
            interrupted requests, outages, quota limits, or failed jobs can leave files
            stored longer, sometimes beyond a day.
          </p>
          <p>
            Individual authorization records expire 48 hours after their last state update.
            Per-IP security counters expire after up to ten minutes or 24 hours; global
            monthly counters contain no IP identifiers and expire after 35 days. Deletion
            and in-memory processing do not guarantee immediate or permanent erasure of
            every copy. Provider caches, buffering, backups, logs, and infrastructure
            retention remain subject to provider practices and hosting configuration.
          </p>
        </section>

        <section aria-labelledby="privacy-security">
          <h2 id="privacy-security">Security</h2>
          <p>
            The application limits image sizes and formats and validates private-upload
            images before inference. Private uploads use short-lived, restricted upload
            tokens and single-use inference authorizations. Storage credentials are kept
            server-side; HTTPS is used for production transfers. These precautions reduce some
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
