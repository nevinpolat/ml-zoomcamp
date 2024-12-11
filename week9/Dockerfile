
FROM public.ecr.aws/lambda/python:3.10

COPY curly-vs-straight.tflite .
COPY lambda_function.py .

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
# Install the required Python packages with compatible versions
RUN pip install numpy==1.23.5 pillow tensorflow==2.14.0 requests

CMD [ "lambda_function.lambda_handler" ]
