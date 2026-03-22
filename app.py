<!DOCTYPE html>
<html>
<head>
  <title>Detector de Presencia</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
  <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest"></script>
</head>
<body>

  <h1>Detector de Persona en Cámara</h1>
  <button onclick="init()">Iniciar Cámara</button>
  
  <div id="webcam-container"></div>
  <div id="label-container"></div>

  <script>
    const URL = "TU_MODELO_URL"; // ← pega aquí tu link

    let model, webcam, labelContainer, maxPredictions;

    async function init() {
      const modelURL = URL + "model.json";
      const metadataURL = URL + "metadata.json";

      model = await tmImage.load(modelURL, metadataURL);
      maxPredictions = model.getTotalClasses();

      webcam = new tmImage.Webcam(300, 300, true);
      await webcam.setup();
      await webcam.play();
      window.requestAnimationFrame(loop);

      document.getElementById("webcam-container").appendChild(webcam.canvas);
      labelContainer = document.getElementById("label-container");
    }

    async function loop() {
      webcam.update();
      await predict();
      window.requestAnimationFrame(loop);
    }

    async function predict() {
      const prediction = await model.predict(webcam.canvas);

      let probPersona = 0;

      for (let i = 0; i < prediction.length; i++) {
        if (prediction[i].className === "Persona") {
          probPersona = prediction[i].probability;
        }
      }

      let resultado = "";

      // 🔥 Aquí está la solución (umbral)
      if (probPersona > 0.90) {
        resultado = "🟢 Estás en cámara (" + (probPersona * 100).toFixed(2) + "%)";
      } else {
        resultado = "🔴 No estás en cámara";
      }

      labelContainer.innerHTML = resultado;
    }
  </script>

</body>
</html>
