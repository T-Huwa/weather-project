// Header scroll effect
document.addEventListener("DOMContentLoaded", function () {
  const header = document.querySelector(".header");
  let lastScrollTop = 0;
  let ticking = false;

  function updateHeader() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    // Add scrolled class when user scrolls down more than 80px
    if (scrollTop > 80) {
      header.classList.add("scrolled");
    } else {
      header.classList.remove("scrolled");
    }

    lastScrollTop = scrollTop;
    ticking = false;
  }

  function requestTick() {
    if (!ticking) {
      requestAnimationFrame(updateHeader);
      ticking = true;
    }
  }

  // Add scroll event listener with throttling for better performance
  window.addEventListener("scroll", requestTick, { passive: true });

  // Also trigger on page load in case user refreshes while scrolled
  updateHeader();
});
