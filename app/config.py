'''BBC 기사 분류 RNN 프로젝트 설정 파일.'''

from dataclasses import dataclass

'''
@dataclass 장식자 (decorator : 데코레이터)
데이터를 저장하기 위한 클래스에 주로 사용함

일반적인 class 작성 코드:
class 클래스명:
    def __init__(self) -> None:
        self.필드명 = 초기값
        ........
    def __repr__(self) -> str:
        # 저장된 필드값들을 하나의 문자열(문장)로 만들어서 출력하는 메소드
        ........
    def __eq__(self, other: object) -> bool:
        # 다른 Config 객체 안의 필드값들과 이 객체 안의 필드값들이 모두 일치하는지 확인하는 메소드
        if isinstance(other, Config):
            return self.필드명 == other.필드명 and .........

=> 클래스 이름 위에 @dataclass 표시하면 
init(), repr(), eq() 를 자동 생성해 주는 데코레이터임
'''

@dataclass
class Config:
    '''학습과 예측에 공통으로 사용하는 하이퍼 파라미터를 한 곳에서 관리하는 클래스.'''

    max_vocab: int = 5000
    max_len: int = 80
    embed_dim: int = 64
    # embed_dim: int = 128
    # hidden_dim: int = 64
    hidden_dim: int = 128
    # batch_size: int = 8
    batch_size: int = 16
    # epochs: int = 8
    epochs: int = 20
    # learning_rate: float = 0.001
    learning_rate: float = 0.0005
    test_size: float = 0.25
    # test_size: float = 0.20
    random_state: int = 42
    model_path: str = './models/bbc_lstm_model.pt'