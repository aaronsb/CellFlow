// Permite ajuste fino de sliders con la rueda del ratón
document.querySelectorAll('input[type=range]').forEach(slider => {
  slider.addEventListener('wheel', function(e) {
    e.preventDefault();
    const step = parseFloat(this.step) || 1;
    let value = parseFloat(this.value);
    let min = parseFloat(this.min);
    let max = parseFloat(this.max);
    // DeltaY: negativo = arriba (incrementar), positivo = abajo (decrementar)
    if (e.deltaY < 0) {
      value += step;
    } else {
      value -= step;
    }
    value = Math.max(min, Math.min(max, value));
    this.value = value;
    // Actualiza el <span> asociado (asume id="xxx-slider" y span id="xxx-value")
    const id = this.id.replace('-slider', '-value');
    const valueSpan = document.getElementById(id);
    if (valueSpan) {
      valueSpan.textContent = step < 1 ? value.toFixed(2) : value;
    }
    // Dispara el evento input para que la simulación reaccione
    this.dispatchEvent(new Event('input', { bubbles: true }));
  });
});
