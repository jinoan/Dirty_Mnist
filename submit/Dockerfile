# cuda 11, pytorch의 베이스 이미지 가져오기
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# opencv-python설치를 위해 libgl1, libgtk2 설치
RUN apt-get update

RUN apt-get install -y libgl1-mesa-glx

RUN apt-get install -y libgtk2.0-dev

# host의 submit폴더를 컨테이너의 /submit로 복사
ADD submit /submit

# workingdir를 /submit으로 설정
WORKDIR /submit

# pip upgrade
RUN python -m pip install --upgrade pip

# library import
RUN python -m pip install -r input/requirements.txt

