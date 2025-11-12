document.addEventListener("DOMContentLoaded", function () {
    const articleUUID = window.articleUUID || null;
    const userID = window.userID || null;

    if (!userID) {
        console.warn("Tracking: Missing userID.");
        return;
    }

    // --- helper: send event and remove null/empty fields ---
    function sendEvent(payload) {
        const cleanPayload = Object.fromEntries(
            Object.entries(payload).filter(([_, v]) => v !== null && v !== undefined && v !== "")
        );
        try {
            navigator.sendBeacon("/track", JSON.stringify(cleanPayload));
        } catch (err) {
            console.error("Tracking error:", err);
        }
    }

    // --- 1) View event ---
    if (articleUUID) {
        sendEvent({
            user_id: userID,
            article_uuid: articleUUID,
            event: "view"
        });
    }

    // --- 2) Scroll progress tracking ---
    let maxScroll = 0;
    let lastSentScroll = 0;

    const updateScroll = () => {
        let scrolled = (window.scrollY + window.innerHeight) / document.documentElement.scrollHeight;
        scrolled = Math.min(Math.max(scrolled, 0), 1.0); // clamp to [0,1]
        scrolled = parseFloat(scrolled.toFixed(2)); // round to 2 decimals

        const scrollPercent = Math.floor(scrolled * 100);

        if (scrollPercent >= lastSentScroll + 10) {
            lastSentScroll = scrollPercent;
            sendEvent({
                user_id: userID,
                article_uuid: articleUUID,
                event: "scroll_progress",
                scroll_depth: scrolled
            });
        }

        maxScroll = Math.max(maxScroll, scrolled);
    };

    window.addEventListener("scroll", () => requestAnimationFrame(updateScroll));

    // --- 3) Paywall click tracking ---
    document.querySelectorAll(".paywall-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            sendEvent({
                user_id: userID,
                article_uuid: articleUUID,
                event: "paywall_click",
                offer_type: btn.dataset.offerType || null
            });
        });
    });

    // --- 4) Leave event with read_to_end ---
    window.addEventListener("pagehide", () => {
        const readToEnd = maxScroll >= 0.98; // >= 98% scrolled
        sendEvent({
            user_id: userID,
            article_uuid: articleUUID,
            event: "leave",
            scroll_depth: parseFloat(maxScroll.toFixed(2)),
            read_to_end: readToEnd
        });
    });
});
