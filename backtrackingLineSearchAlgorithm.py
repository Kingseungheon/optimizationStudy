#fixed step size alpha = 1/2

import numpy as np #넘파이 불러옴 (꼭  필요하진 않지만 수치계산용 관례임)

def f(x) : #목적함수 정의 (최소화 대상임)
    return x**2 - (1/3)*x**3 

def grad_f(x) : #1차 미분 한거(그래디언트). 1차원이라 스칼라
    return 2*x - x**2

#백트래킹 라인서치 함수 
# x:현재점
# d:탐색방향 (이건 그냥 d)
# alpha:초기 스텝 (보통 1로 둠) (t로 생각하면 됨)
# rho: 줄이는 비율 0 < rho < 1 (베타로 생각하면 됨)
# c: Armijo 상수 0<c<1 (예: 1e-4) 
def backtracking_line_search(f, grad_f, x, d, alpha=1, rho=0.5, c=1e-4): 
    while f(x + alpha * d) > f(x) + c * alpha * grad_f(x) * d: #조건검사 : ppt에서 백트래킹 초기 조건중에 첫번째 만족하는 지
        alpha *= rho #실패 시 스탭을 rho를 곱하여 줄인다
    return alpha #조건을 만족하는 alpha 반환 이때의 알파는 점에서 1차 근사한 선의 값보다 작을 때의 조건에서 베타배 만큼 한 값임

def gradient_descent_backtracking(x0, max_iter=20): #초기 x가 x0인거고 max 반복은 20번 
    x = x0
    for k in range(max_iter):
        g = grad_f(x)                                       # g에 그래디언트 함수를 호출하여 가장 빠르게 증가하는 방향의 값을 집어넣음
        d = -g                                              # 마이너스 그래디언트 => 가장 가파른 하강 방향 => 이거를 direction d에 대입
        alpha = backtracking_line_search(f, grad_f, x, d)   # 주어진 방향 d에서 백트래킹으로 스텝사이즈 alpha 결정
        x = x + alpha * d                                   # 그 다음 x값 즉 다음 스텝으로 한 스텝 이동한거임
        #아래 출력: 현재 진행상태 출력(반복번호(k+1), 현재점(x), 함수값(f(x)), 스텝(alpha))
        #f" ===>>> f"Hello {name}" 이렇게 쓰면 name 변수의 실제값이 문자열 안으로 들어감
        #{}: f-string 파이썬의 서식 문자열 (format-string) 기능
        #{name} 이렇게 쓰면 name을 실제 값으로 출력해준다
        #02d는 정수 출력 형식 0->빈자리는 0으로 채운다 2->최소 두자리로 표시 d->정수
        #x={x:.6f} x의 값을 소수점 아래 6자리까지 표시 7번째에서 반올림
        print(f"iter {k+1:02d} : x = {x:.6f}, f(x) = {f(x):.6f}, alpha = {alpha:.4f}")
        if abs(g) < 1e-6: #만약 이동 전 그래디언트(g)의 크기가 매우 작으면(최적 근처이므로) 중단
            break
    return x

x_min = gradient_descent_backtracking(x0=1.2) #추정된 최소점 반환
print(f"\nEstimated mininizer: x* = {x_min:.6f}")

#rho가 너무 작으면 (Ex)0.1) 스텝이 급격히 줄어 들어 평가가 많아질 수 있음
#c가 너무 크면 (예: 0.5) "충분감소" 기준이 느슨해져 성능 저하 가능
#1D라 단순하지만, 다변수도 동일하되 grad_f(x)와 d가 벡터, 내적은 dot으로 바뀜






'''import numpy as np

x = np.array([[1,2,3],[4,5,6]])

print("x:\n", x)
'''


