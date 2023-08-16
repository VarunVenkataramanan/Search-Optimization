FROM phusion/passenger-full:latest
RUN apt update
RUN apt install pip -y
RUN apt install sudo -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html notebook transformers
RUN pip3 install datasets scikit-learn seqeval evaluate accelerate  
RUN gem install pycall
CMD ["jupyter","notebook","--ip","0.0.0.0","--port","8889","--no-browser","--allow-root"]
