from flask import Flask, render_template
from flask_socketio import SocketIO
from process import handle_start_process

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('start_process')
def start_process():
    handle_start_process(socketio)

if __name__ == '__main__':
    socketio.run(app, debug=True)
