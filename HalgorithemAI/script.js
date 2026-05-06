document.querySelectorAll(".navbar-nav .nav-link").forEach(link => {
  link.addEventListener("click", () => {
    const menu = document.querySelector(".navbar-collapse.show");
    if (!menu) return;

    const collapse = bootstrap.Collapse.getOrCreateInstance(menu);
    collapse.hide();
  });
});
