# port id is always 6006
port_id=$1
gunicorn -w 1 -b 0.0.0.0:$port_id assistent_service:app
