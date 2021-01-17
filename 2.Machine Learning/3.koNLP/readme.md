# 한국어 형태소 분석
## Korean POS tags comparison chart
- 주소 : https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0
- 음영 표시 : koNLP 에서 사용할 수 있는 항목

## 한나눔 매뉴얼
- https://www.sketchengine.eu/wp-content/uploads/Original-HanNanum-manual.pdf

# 기본 작업에 필요한 함수들
- koNLPy
    - Python Version(python --version)에 맞춰서 설치해야 함
    - Anaconda와 연동할 경우
        - python interpriter 별도 설치 말 것
        - Anaconda에서 설치한 python interpriter 사용
        - 이렇게 안하면 100% 충돌
    - 설치방법
        - 기본 설치 : pip install konlpy nltk
        - OS별 설치 Manual : https://konlpy.org/en/latest/install/#
- JRE
    - 설치 후 환경변수(PATH) 꼭 확인
    - 확인 안하면 정말 짜증나는 상황 발생

#wordcloud 실습 함수
- wordcloud : pip install wordcloud
- vscode Extended Module : Jupyter (deprecated)
    - 정의 : Jupyter Notebook 처럼 Graph 띄우기
    - 코드 첫줄에 #%% 삽입
    - 실행 : "Run Cell" 선택(또는 Shift + Enter)
- masking 사용 시 : pip install image numpy np
