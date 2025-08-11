from flask import Flask, render_template, request, jsonify
import os
from google.cloud import vision
import base64
import requests

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', active_page='home')

@app.route('/contacto')
def contacto():
    return render_template('contacto.html', active_page='contacto')

@app.route('/precios')
def precios():
    return render_template('precios.html', active_page='precios')

@app.route('/ayuda')
def ayuda():
    return render_template('ayuda.html', active_page='ayuda')


# Ruta para analizar imagen usando Google Vision API
@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    data = request.get_json()
    base64_image = data.get('image')
    try:
        # Configurar credenciales
        cred_path = os.path.join(os.path.dirname(__file__), 'vertex-credentials.json')
        client = vision.ImageAnnotatorClient.from_service_account_file(cred_path)
        image = vision.Image(content=base64.b64decode(base64_image))
        response = client.web_detection(image=image)
        web_detection = response.web_detection
        result_text = ""
        if web_detection.web_entities:
            entidades = [entity.description for entity in web_detection.web_entities if entity.description]
            result_text += f"Entidades detectadas: {', '.join(entidades)}. "
        if web_detection.best_guess_labels:
            guess = ', '.join([label.label for label in web_detection.best_guess_labels])
            result_text += f"Mejor suposición: {guess}. "
        if result_text:
            return jsonify({"result": result_text.strip()})
        else:
            return jsonify({"result": "No se pudo identificar la raza ni entidades relevantes en la imagen."})
    except Exception as e:
        print(e)
        return jsonify({"result": "Hubo un error al analizar la imagen."}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)