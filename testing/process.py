from flask_socketio import emit

class handle_start_process:
    def __init__(self, socketio):
        self.socketio = socketio
        self.start_process()

    def start_process(self):
        print('Process started')
        
        # Extracting data
        emit('status_update', {'status': 'Extracting data...', 'step': 'extracting'})
        self.socketio.sleep(2)  # Simulate time-consuming process
        emit('status_update', {'status': 'Extracting data...Done', 'step': 'extracting'})
        
        # Analyzing data
        emit('status_update', {'status': 'Analyzing data...', 'step': 'analyzing'})
        self.socketio.sleep(2)
        emit('status_update', {'status': 'Analyzing data...Done', 'step': 'analyzing'})
        
        # Separating PDF
        emit('status_update', {'status': 'Separating PDF...', 'step': 'separating'})
        self.socketio.sleep(2)
        emit('status_update', {'status': 'Separating PDF...Done', 'step': 'separating'})
        
        print('Process completed')