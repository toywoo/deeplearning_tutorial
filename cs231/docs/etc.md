### range() vs xrange
함수의 기능은 같지만 타입이 다르다. range()는 list이고 xrange()는 xrange 타입이다. xrange는 수정이 불가한 순차적 접근 가능한 데이터 타입으로 메모리 할당량이 일정하다. 그래서 지정하는 범위가 커지면 커질 수록 메모리 사용 효율이 좋아지게 된다 늘어났을 때 새롭게 할당을 하지 않아도 되니까.

Nearest Neighbor를 더 빠르게 하는 법
facebook Faiss를 참고하자.