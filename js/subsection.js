/**
 * Weather Subscription Functionality
 * Pure JavaScript implementation for subscription form handling
 * No rendering - only functionality and modal management
 */

(function () {
  "use strict";

  // Configuration
  const config = {
    apiBaseUrl: "https://weather-project-orjd.onrender.com",
    endpoints: {
      subscribe: "/subscribe",
      subscriberCount: "/subscribers/count",
    },
    selectors: {
      form: "#weatherSubscriptionForm",
      emailInput: "#weatherEmail",
      phoneInput: "#weatherPhone",
      submitButton: "#subscribeButton",
      buttonText: ".weather-sub-button-text",
      spinner: "#submitSpinner",
      subscriberCount: "#subscriberCount",
      modal: "#weatherSubscriptionModal",
      modalBackdrop: "#weatherModalBackdrop",
      modalIcon: "#modalIcon",
      modalTitle: "#modalTitle",
      modalMessage: "#modalMessage",
      modalButton: "#modalButton",
      modalClose: "#modalClose",
    },
  };

  // Weather Subscription Class
  class WeatherSubscription {
    constructor() {
      this.elements = {};
      this.isSubmitting = false;
      this.init();
    }

    // Initialize the subscription functionality
    init() {
      this.cacheElements();
      this.bindEvents();
      this.loadSubscriberCount();
    }

    // Cache DOM elements
    cacheElements() {
      Object.keys(config.selectors).forEach((key) => {
        this.elements[key] = document.querySelector(config.selectors[key]);
      });
    }

    // Bind event listeners
    bindEvents() {
      // Form submission
      if (this.elements.form) {
        this.elements.form.addEventListener(
          "submit",
          this.handleSubmit.bind(this)
        );
      }

      // Email validation
      if (this.elements.emailInput) {
        this.elements.emailInput.addEventListener(
          "blur",
          this.validateEmail.bind(this)
        );
        this.elements.emailInput.addEventListener(
          "input",
          this.clearEmailError.bind(this)
        );
      }

      // Phone number cleaning
      if (this.elements.phoneInput) {
        this.elements.phoneInput.addEventListener(
          "input",
          this.cleanPhoneNumber.bind(this)
        );
      }

      // Modal controls
      if (this.elements.modalClose) {
        this.elements.modalClose.addEventListener(
          "click",
          this.closeModal.bind(this)
        );
      }
      if (this.elements.modalButton) {
        this.elements.modalButton.addEventListener(
          "click",
          this.closeModal.bind(this)
        );
      }
      if (this.elements.modalBackdrop) {
        this.elements.modalBackdrop.addEventListener(
          "click",
          this.closeModal.bind(this)
        );
      }

      // Keyboard events
      document.addEventListener("keydown", this.handleKeydown.bind(this));
    }

    // Handle form submission
    async handleSubmit(event) {
      event.preventDefault();

      if (this.isSubmitting) return;

      const email = this.getCleanEmail();
      const phone = this.getCleanPhone();

      // Validate email
      if (!this.isValidEmail(email)) {
        this.showEmailError("Please enter a valid email address.");
        this.elements.emailInput.focus();
        return;
      }

      // Start submission
      this.isSubmitting = true;
      this.showLoading();

      try {
        const response = await this.submitSubscription(email, phone);

        if (response.success) {
          this.handleSuccess(response.data);
        } else {
          this.handleError(response.error);
        }
      } catch (error) {
        console.error("Subscription error:", error);
        this.handleError(
          "Network error. Please check your connection and try again."
        );
      } finally {
        this.isSubmitting = false;
        this.hideLoading();
      }
    }

    // Submit subscription to API
    async submitSubscription(email, phone) {
      try {
        const response = await fetch(
          `${config.apiBaseUrl}${config.endpoints.subscribe}`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              email: email,
              phone_number: phone || null,
            }),
          }
        );

        const data = await response.json();

        return {
          success: response.ok,
          data: data,
          error: response.ok
            ? null
            : data.error || "Something went wrong. Please try again.",
        };
      } catch (error) {
        throw new Error("Failed to connect to server");
      }
    }

    // Handle successful subscription
    handleSuccess(data) {
      this.resetForm();
      this.showModal(
        "success",
        "Welcome Aboard! 🎉",
        "Thank you for subscribing to weather predictions! You'll receive your first forecast on the 1st of next month. Check your inbox for a welcome email."
      );

      // Update subscriber count after a short delay
      setTimeout(() => this.loadSubscriberCount(), 1500);

      // Track analytics if available
      this.trackEvent("subscribe", "success");
    }

    // Handle subscription error
    handleError(errorMessage) {
      this.showModal(
        "error",
        "Subscription Failed",
        errorMessage ||
          "We couldn't process your subscription right now. Please try again in a few minutes."
      );

      this.trackEvent("subscribe", "error");
    }

    // Load and display subscriber count
    async loadSubscriberCount() {
      if (!this.elements.subscriberCount) return;

      try {
        const response = await fetch(
          `${config.apiBaseUrl}${config.endpoints.subscriberCount}`
        );

        if (response.ok) {
          const data = await response.json();
          this.animateCounterUpdate(data.active_subscribers);
        } else {
          this.elements.subscriberCount.textContent = "100+";
        }
      } catch (error) {
        console.log("Could not load subscriber count:", error);
        this.elements.subscriberCount.textContent = "100+";
      }
    }

    // Animate counter update
    animateCounterUpdate(newCount) {
      const element = this.elements.subscriberCount;
      const currentCount = parseInt(element.textContent) || 0;

      if (currentCount === newCount) return;

      const duration = 1000;
      const steps = 30;
      const increment = (newCount - currentCount) / steps;
      let current = currentCount;
      let step = 0;

      const timer = setInterval(() => {
        step++;
        current += increment;

        if (step >= steps) {
          element.textContent = newCount;
          clearInterval(timer);
        } else {
          element.textContent = Math.round(current);
        }
      }, duration / steps);
    }

    // Show loading state
    showLoading() {
      if (this.elements.submitButton) {
        this.elements.submitButton.disabled = true;
      }
      if (this.elements.buttonText) {
        this.elements.buttonText.textContent = "Subscribing...";
      }
      if (this.elements.spinner) {
        this.elements.spinner.style.display = "block";
      }
    }

    // Hide loading state
    hideLoading() {
      if (this.elements.submitButton) {
        this.elements.submitButton.disabled = false;
      }
      if (this.elements.buttonText) {
        this.elements.buttonText.textContent = "Subscribe to Predictions";
      }
      if (this.elements.spinner) {
        this.elements.spinner.style.display = "none";
      }
    }

    // Reset form
    resetForm() {
      if (this.elements.form) {
        this.elements.form.reset();
      }
      this.clearEmailError();
    }

    // Email validation
    isValidEmail(email) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      return emailRegex.test(email);
    }

    // Get clean email
    getCleanEmail() {
      return this.elements.emailInput
        ? this.elements.emailInput.value.trim().toLowerCase()
        : "";
    }

    // Get clean phone
    getCleanPhone() {
      return this.elements.phoneInput
        ? this.elements.phoneInput.value.trim()
        : "";
    }

    // Validate email on blur
    validateEmail(event) {
      const email = event.target.value.trim();

      if (email && !this.isValidEmail(email)) {
        this.showEmailError("Please enter a valid email address.");
      } else {
        this.clearEmailError();
      }
    }

    // Show email error
    showEmailError(message) {
      if (this.elements.emailInput) {
        this.elements.emailInput.classList.add("error");
        this.elements.emailInput.classList.remove("success");
      }
    }

    // Clear email error
    clearEmailError() {
      if (this.elements.emailInput) {
        this.elements.emailInput.classList.remove("error");

        const email = this.elements.emailInput.value.trim();
        if (email && this.isValidEmail(email)) {
          this.elements.emailInput.classList.add("success");
        } else {
          this.elements.emailInput.classList.remove("success");
        }
      }
    }

    // Clean phone number input
    cleanPhoneNumber(event) {
      const cleaned = event.target.value.replace(/[^\d+\-\s\(\)]/g, "");
      event.target.value = cleaned;
    }

    // Show modal
    showModal(type, title, message) {
      if (!this.elements.modal || !this.elements.modalBackdrop) return;

      // Set modal content
      if (this.elements.modalIcon) {
        this.elements.modalIcon.textContent = type === "success" ? "✓" : "⚠";
        this.elements.modalIcon.className = `weather-modal-icon ${type}`;
      }

      if (this.elements.modalTitle) {
        this.elements.modalTitle.textContent = title;
      }

      if (this.elements.modalMessage) {
        this.elements.modalMessage.textContent = message;
      }

      // Show modal
      this.elements.modalBackdrop.classList.add("show");
      this.elements.modal.classList.add("show");

      // Focus management
      this.trapFocus();

      // Auto-close success modals after 5 seconds
      if (type === "success") {
        setTimeout(() => {
          if (this.elements.modal.classList.contains("show")) {
            this.closeModal();
          }
        }, 5000);
      }
    }

    // Close modal
    closeModal() {
      if (!this.elements.modal || !this.elements.modalBackdrop) return;

      this.elements.modal.classList.remove("show");
      this.elements.modalBackdrop.classList.remove("show");

      // Return focus to email input
      if (this.elements.emailInput) {
        this.elements.emailInput.focus();
      }
    }

    // Trap focus within modal
    trapFocus() {
      if (!this.elements.modal) return;

      const focusableElements = this.elements.modal.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );

      if (focusableElements.length === 0) return;

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      // Focus first element
      firstElement.focus();

      // Handle tab key
      const handleTab = (e) => {
        if (e.key !== "Tab") return;

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      };

      document.addEventListener("keydown", handleTab);

      // Remove event listener when modal closes
      const removeListener = () => {
        document.removeEventListener("keydown", handleTab);
        document.removeEventListener("keydown", removeListener);
      };

      // Clean up when modal is closed
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (
            mutation.type === "attributes" &&
            mutation.attributeName === "class" &&
            !this.elements.modal.classList.contains("show")
          ) {
            removeListener();
            observer.disconnect();
          }
        });
      });

      observer.observe(this.elements.modal, { attributes: true });
    }

    // Handle keyboard events
    handleKeydown(event) {
      // Close modal on Escape key
      if (
        event.key === "Escape" &&
        this.elements.modal &&
        this.elements.modal.classList.contains("show")
      ) {
        event.preventDefault();
        this.closeModal();
      }

      // Submit form on Ctrl+Enter
      if (
        (event.ctrlKey || event.metaKey) &&
        event.key === "Enter" &&
        (event.target === this.elements.emailInput ||
          event.target === this.elements.phoneInput)
      ) {
        event.preventDefault();
        this.elements.form.dispatchEvent(new Event("submit"));
      }
    }

    // Track analytics events
    trackEvent(action, result) {
      // Google Analytics 4
      if (typeof gtag !== "undefined") {
        gtag("event", action, {
          event_category: "weather_subscription",
          event_label: result,
          value: 1,
        });
      }

      // Google Analytics Universal
      if (typeof ga !== "undefined") {
        ga("send", "event", "weather_subscription", action, result, 1);
      }

      // Facebook Pixel
      if (
        typeof fbq !== "undefined" &&
        action === "subscribe" &&
        result === "success"
      ) {
        fbq("track", "Subscribe");
      }

      // Custom analytics
      if (typeof window.customAnalytics === "function") {
        window.customAnalytics("weather_subscription", action, result);
      }
    }

    // Public method to update configuration
    updateConfig(newConfig) {
      Object.assign(config, newConfig);
    }

    // Public method to reload subscriber count
    reloadSubscriberCount() {
      this.loadSubscriberCount();
    }

    // Public method to show custom modal
    showCustomModal(type, title, message) {
      this.showModal(type, title, message);
    }

    // Public method to programmatically subscribe
    async programmaticSubscribe(email, phone = null) {
      if (!this.isValidEmail(email)) {
        throw new Error("Invalid email address");
      }

      try {
        const response = await this.submitSubscription(email, phone);

        if (response.success) {
          this.trackEvent("subscribe", "success");
          return { success: true, data: response.data };
        } else {
          this.trackEvent("subscribe", "error");
          throw new Error(response.error);
        }
      } catch (error) {
        throw new Error(`Subscription failed: ${error.message}`);
      }
    }

    // Destroy instance and clean up
    destroy() {
      // Remove event listeners
      if (this.elements.form) {
        this.elements.form.removeEventListener("submit", this.handleSubmit);
      }
      if (this.elements.emailInput) {
        this.elements.emailInput.removeEventListener(
          "blur",
          this.validateEmail
        );
        this.elements.emailInput.removeEventListener(
          "input",
          this.clearEmailError
        );
      }
      if (this.elements.phoneInput) {
        this.elements.phoneInput.removeEventListener(
          "input",
          this.cleanPhoneNumber
        );
      }

      // Clear elements cache
      this.elements = {};
      this.isSubmitting = false;
    }
  }

  // Auto-initialize when DOM is ready
  function initializeWhenReady() {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => {
        if (document.querySelector(config.selectors.form)) {
          window.weatherSubscription = new WeatherSubscription();
        }
      });
    } else {
      if (document.querySelector(config.selectors.form)) {
        window.weatherSubscription = new WeatherSubscription();
      }
    }
  }

  // Public API
  window.WeatherSubscriptionAPI = {
    // Initialize manually
    init: function (customConfig = {}) {
      if (customConfig) {
        Object.assign(config, customConfig);
      }
      window.weatherSubscription = new WeatherSubscription();
      return window.weatherSubscription;
    },

    // Get current instance
    getInstance: function () {
      return window.weatherSubscription || null;
    },

    // Update configuration
    updateConfig: function (newConfig) {
      Object.assign(config, newConfig);
      if (window.weatherSubscription) {
        window.weatherSubscription.updateConfig(newConfig);
      }
    },

    // Destroy current instance
    destroy: function () {
      if (window.weatherSubscription) {
        window.weatherSubscription.destroy();
        window.weatherSubscription = null;
      }
    },
  };

  // Auto-initialize
  initializeWhenReady();

  // Debug information
  if (
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
  ) {
    console.log("Weather Subscription API loaded");
    console.log(
      "Available methods:",
      Object.keys(window.WeatherSubscriptionAPI)
    );
    console.log("Current config:", config);
  }
})();
