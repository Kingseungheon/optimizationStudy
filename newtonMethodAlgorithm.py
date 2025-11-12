# hessian이 나쁘게 나올 수도 있으므로, 백트래킹 라인 서치를 도입해야 될 수도 있음
import numpy as np 

def f(x): 
    return (x[0]+ 10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10*(x[0]-x[3])**4

def grad_f(x): 
    return np.array ([2*(x[0] + 10 * x[1]) + 40 * (x[0] - x[3]) ** 3, 20*(x[0] + 10*x[1]) + 4 * (x[1] - 2*x[2]) ** 3, 10*(x[2]-x[3]) - 8*(x[1]-2*x[2])**3,  -10 * (x[2] - x[3]) - 40 * (x[0]-x[3])**3])

def hess_f(x): 
    return np.array([[2+120*(x[0]-x[3])**2, 20, 0, -120*(x[0]-x[3])**2],
                     [20, 200+12*(x[1]-2*x[2])**2, -24*(x[1]-2*x[2])**2, 0], 
                     [0, -24*(x[1]-2*x[2])**2, 10+48*(x[1]-2*x[2])**2, -10], 
                     [-120*(x[0]-x[3])**2, 0, -10, 10+120*(x[0]-x[3])**2]
                     ])

def newton_method_multivar(x0, tol=1e-10, max_iter=50):
    x = x0.astype(float) 
    for i in range(max_iter): 
        grad = grad_f(x) 
        hess = hess_f(x) 
        #x_new = x - np.linalg.inv(hess) @ grad #뉴턴 업데이트 k+1번째 x = k번째 x - k번째 헤시안 분의 f(x_k)
        x_new = x - np.linalg.solve(hess, grad) #-> 수치 안정성/ 성능이 더 좋음
        #s = np.linalg.solve(hess, grad) #헤시안 역행렬과 그래디언트의 곱을 s에 넣는 것
        
        #if np.linalg.norm(x_new - x) < tol: #x_new와 x의 차이가 tol보다 작으면 수렴
        if np.linalg.norm(grad) < tol: #그래디언트의 크기가 tol보다 작으면 수렴
            print(f"Converged in {i+1} iterations.") 
            return x_new 
        x = x_new 
    print("Did not converge.")  
    return x        
        
x_star = newton_method_multivar(np.array([3.0, -1.0, 0.0, 1.0]))  #초기값이 무엇이든 x_1에서수렴함
print("Minimizer:", x_star) #도출된 최소점과 그 위치에서의 함수값 출력
print("Minimum value f(x*):", f(x_star)) #f(x*) = 0.0


'''

# hessian이 상수 PD, f가 정확한 이차형(quadratic)임
# newton method -> 2차 근사를 사용 if 실제 함수가 2차면 2차 근사가 정확.
# ==> 첫 스텝에서 꼭짓점(최소점)을 바로 찾아감.

import numpy as np #배열, 전치, 내적, 역행렬, 노름 등에 사용됨

def f(x): #스칼라 값을 반환하는 함수
    return (x[0]-1)**2 + 2*(x[1]-2)**2 #(x_1-2)^2 + 2(x_2-2)^2  /x의 변수 종류가 두개인 거임.

def grad_f(x): #편미분들을 모아 기울기 벡터를 생성: x_1에 대해 미분, x_2에 대해 미분 => 길이 2의 numpy배열
    return np.array([2*(x[0]-1), 4*(x[1]-2)])

def hess_f(x): #헤시안 2x2 행렬
    return np.array([[2, 0],[0, 4]])

#뉴턴 메서드 본체
#x0 : 시작점
#tol : 수렴 허용오차(스텝 변화량 기준)
#max_iter : 최대 반복 횟수 안전장치
#x는 현재 추정치(iterate)
def newton_method_multivar(x0, tol=1e-6, max_iter=50):
    x = x0.astype(float) #입력 시작점을 실수형 numpy 배열로 변환
    hess = hess_f(x) #현재 점 x에서 Hessian 계산
    for i in range(max_iter): #50번 반복하겠다
        grad = grad_f(x) #현재 점 x에서 gradient 계산
        
        # @는 행렬, 벡터 곱
        # 이 예제는 H가 상수 PD라서 역행렬이 항상 존재, 스텝이 깔끔함
        
        
        #뉴턴 업데이트 식
        #linalg.inv(): 역행렬 계산 
        #hess: 현재 점에서의 헤시안 행렬
        #@: 행렬 곱셈 연산자
        #grad: 현재 점에서의 그래디언트 벡터
        #
        #x_new = x - np.linalg.inv(hess) @ grad #뉴턴 업데이트 k+1번째 x = k번째 x - k번째 헤시안 분의 f(x_k)
        x_new = x - np.linalg.solve(hess, grad) #-> 수치 안정성/ 성능이 더 좋음
        #s = np.linalg.solve(hess, grad) #헤시안 역행렬과 그래디언트의 곱을 s에 넣는 것
        if np.linalg.norm(x_new - x) < tol: #수렴판정 : 새 점과 이전 점의 차이(유클리드 노름)가 tol보다 작으면 수렴으로 간주함
            print(f"Converged in {i+1} iterations.") #i+1번째 즉 i는 0번에서 시작하므로 i+1번째 반복에서 수렴한다 (몇 번 반복했는지 출력)
            return x_new #수렴할 경우 최소점 후보 x_new를 반환
        x = x_new #다음 반복을 위해 현재 점 갱신.
        
    #제한 횟수 내에 수렴 실패 시 메시지 출력 후 마지막 추정값 반환
    print("Did not converge.")  
    return x        


#시작점 x_0 = (0, 0)뉴턴 메서드 실행 -> 이 문제는 진짜 2차 함수이고 헤시안이 상수 PD-> 한 번에 수렴(exact quadratic이면 이론적으로 1스텝)        
x_star = newton_method_multivar(np.array([0.0, 0.0]))  #초기값이 무엇이든 x_1에서수렴함
print("Minimizer:", x_star) #도출된 최소점과 그 위치에서의 함수값 출력
print("Minimum value f(x*):", f(x_star)) #f(x*) = 0.0
'''