FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 7312
CMD ["streamlit", "run", "app.py", "--server.port=7312", "--server.address=0.0.0.0"]
