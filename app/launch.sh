export FLASK_DEBUG=1
gunicorn --bind 0.0.0.0:8081 app:app
