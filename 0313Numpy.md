# Numpy 개념 필사

## 머신러닝
    머신러닝: 애플리케이션을 수정하지 않고도 데이터를 기반을 패턴을 학습하고 결과를 예측하는 알고리즘 기법
    머신러닝=지도학습+비지도학습
## Numpy
    numpy의 기반 데이터 타입: ndarray
    ndarray의 데이터값은 문자열,숫자,bool 모두 가능, 하지만 ndarray내 데이터 타입은 같은 데이터 타입만 가능
    (리스트와 다르게)
    만약 서로 다른 데이터 유형이 섞여있다면 더 큰 데이터 타입으로 형 변환을 일괄 적용함
- type(array): class'numpy.ndarray'
- array.shape: (row,col) #행과 열의 수를 튜플 형태로 가지고 있어 ndarray 배열의 차원을 알 수 있음
(n,): 1차원, (n,m): 2차원
- array.ndim:리스트[]=1차원, 리스트[[]]=2차원
- np.arange(n):0~(n-1) 까지의 값을 순차적으로 ndarray의 데이터값으로 변환시켜줌, start=a,stop=b값 추가 가능
- np.zeros((n,p),dtype='int32'): 모든 값을 0으로 채움
- np.ones((n,p)): 모든 값을 1로 채움, 디폴트 값은 float64
- array.reshape(n,p): 차원과 크기 변환시켜줌, -1을 인자로 사용하면 나머지 값에 맞춰 자동배정
- array.tolist(): 리스트 자료형태로 print
- array[0,3]: row=0,col=3인 값 추출하기
- array[:4,3:4]:row=0,1,2,3, col=3인 array 추출하기
- array[0]: 첫번째 row의 값들 추출하기,리스트형태
- array[3]: 네번째 row의 값들 추출하기,리스트형태
- array[[0,1],3]:네번째 col의 첫번째, 두번째 값 리스트로 리턴하기
- array[조건]: 조건이 true가 되는 값이 리턴됨
- np.sort(array): 이때 원본 행렬을 sort된 행렬이 반환됨, 원본행렬을 변하지 않음
- array.sort(): 이때 반환되는 값 없음, 원본행렬이 sort된 행렬로 대체됨
- np.sort(array)[::-1]: 내림차순으로 정렬하기
- np.sort(array,axis=0): row를 오름차순으로 정리해줌
- np.sort(array,axis=1): column을 오름차순으로 정리해줌
- np.argsort(array): 원본행렬 정렬시 index값을 리턴
- np.dot(A,B): A,B 행렬의 내적 결과 알려줌
- np.transpose(A): A행렬의 전치행렬


```python
import numpy as np
```


```python
array1=np.array([1,2,3,4,5])
print('array1 type',type(array1))
print('array1 array 형태',array1.shape)
#1차원
```

    array1 type <class 'numpy.ndarray'>
    array1 array 형태 (5,)
    


```python
array2=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print('array2 type',type(array2))
print('array2 array 형태',array2.shape)
#2차원
```

    array2 type <class 'numpy.ndarray'>
    array2 array 형태 (2, 5)
    


```python
array3=np.array([[1,2,3,4,5]])
print('array3 type',type(array3))
print('array3 array 형태',array3.shape)
#3차원
```

    array3 type <class 'numpy.ndarray'>
    array3 array 형태 (1, 5)
    


```python
list1=[1,2,3]
print(type(list1))
array1=np.array(list1)
print(type(array1))
print(array1,array1.dtype)
```

    <class 'list'>
    <class 'numpy.ndarray'>
    [1 2 3] int32
    


```python
list2=[1,2,3,'test']
print(type(list2))
array2=np.array(list2)
print(type(array2))
print(array2,array2.dtype)
```

    <class 'list'>
    <class 'numpy.ndarray'>
    ['1' '2' '3' 'test'] <U11
    


```python
list3=[1,2,3,4.56]
print(type(list3))
array3=np.array(list3)
print(type(array3))
print(array3,array3.dtype)
```

    <class 'list'>
    <class 'numpy.ndarray'>
    [1.   2.   3.   4.56] float64
    


```python
intarray=np.array([1,2,3,4])
floatarray=intarray.astype('float64')
intarray2=floatarray.astype('int32')
print(intarray,intarray.dtype)
print(floatarray,floatarray.dtype)
print(intarray2,intarray2.dtype)
```

    [1 2 3 4] int32
    [1. 2. 3. 4.] float64
    [1 2 3 4] int32
    


```python
sequence_array=np.arange(10)
print(sequence_array)
print(sequence_array.dtype)
print(sequence_array.shape)
```

    [0 1 2 3 4 5 6 7 8 9]
    int32
    (10,)
    


```python
zero_array=np.zeros((3,2),dtype='int32')
print(zero_array)
print(zero_array.dtype,zero_array.shape)
```

    [[0 0]
     [0 0]
     [0 0]]
    int32 (3, 2)
    


```python
one_array=np.ones((5,3))
print(one_array)
print(one_array.dtype,one_array.shape)
```

    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    float64 (5, 3)
    


```python
array1=np.arange(10)
print('array1:\n',array1)
array2=array1.reshape(2,5)
print('array2:\n',array2)
array3=array1.reshape(5,2)
print('array3:\n',array3)
```

    array1:
     [0 1 2 3 4 5 6 7 8 9]
    array2:
     [[0 1 2 3 4]
     [5 6 7 8 9]]
    array3:
     [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]
    


```python
array1=np.arange(12)
array2=array1.reshape(-1,4)
array3=array2.reshape(2,-1)
print('array1:\n',array1)
print('array2:\n',array2)
print('array3:\n',array3)
```

    array1:
     [ 0  1  2  3  4  5  6  7  8  9 10 11]
    array2:
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    array3:
     [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]]
    


```python
array1=np.arange(8)
array3d=array1.reshape((2,2,2))
print('array3d:\n',array3d)
print(array3d.tolist())
array2=array3d.reshape((-1,2))
print('array2:\n',array2.tolist())
print(array2.shape)
```

    array3d:
     [[[0 1]
      [2 3]]
    
     [[4 5]
      [6 7]]]
    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    array2:
     [[0, 1], [2, 3], [4, 5], [6, 7]]
    (4, 2)
    


```python
array1=np.arange(start=1,stop=10)
print('array1:',array1)
val=array1[3]
print('value:',val)
print(type(val))
```

    array1: [1 2 3 4 5 6 7 8 9]
    value: 4
    <class 'numpy.int32'>
    


```python
array1[0]=10
array1[-1]=11
print('array1:',array1)
```

    array1: [10  2  3  4  5  6  7  8 11]
    


```python
array1d=np.arange(start=1,stop=10)
array2d=array1d.reshape(3,3)
print(array2d)
print('row=0,col=0 의 값',array2d[0,0])
print('row=0,col=2 의 값',array2d[0,2])
print('row=2,col=0 의 값',array2d[2,0])
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    row=0,col=0 의 값 1
    row=0,col=2 의 값 3
    row=2,col=0 의 값 7
    


```python
array1=np.arange(start=1,stop=13)
array2=array1[3:7]
print(array2)
print(type(array2))
print(array1[:6])
print(array1[4:])
print(array1[:])
```

    [4 5 6 7]
    <class 'numpy.ndarray'>
    [1 2 3 4 5 6]
    [ 5  6  7  8  9 10 11 12]
    [ 1  2  3  4  5  6  7  8  9 10 11 12]
    


```python
array1d=np.arange(start=1,stop=13)
array2d=array1.reshape(6,2)
print('array2d:\n',array2d)
print('array2d[1:3,0:1]:\n',array2d[1:3,0:1])
print('array2d[:2,:]:\n',array2d[:2,:])
print('array2d[:,:]:\n',array2d[:,:])
```

    array2d:
     [[ 1  2]
     [ 3  4]
     [ 5  6]
     [ 7  8]
     [ 9 10]
     [11 12]]
    array2d[1:3,0:1]:
     [[3]
     [5]]
    array2d[:2,:]:
     [[1 2]
     [3 4]]
    array2d[:,:]:
     [[ 1  2]
     [ 3  4]
     [ 5  6]
     [ 7  8]
     [ 9 10]
     [11 12]]
    


```python
print(array2d[0])
print(array2d[2])
```

    [1 2]
    [5 6]
    


```python
array1d=np.arange(start=1,stop=10)
array2d=array1d.reshape(3,3)
array3=array2d[[0,1],2]
print(array2d)
print('array2d[[0,1],2]:\n',array3.tolist())
array4=array2d[[0,2],0:2]
print('array2d[[0,2],0:2]:\n',array4,'\n',array4.tolist())
array5=array2d[[0,2]]
print('array2d[[0,2]]:\n',array5,'\n',array5.tolist())
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    array2d[[0,1],2]:
     [3, 6]
    array2d[[0,2],0:2]:
     [[1 2]
     [7 8]] 
     [[1, 2], [7, 8]]
    array2d[[0,2]]:
     [[1 2 3]
     [7 8 9]] 
     [[1, 2, 3], [7, 8, 9]]
    


```python
array3=array1d[array1d>5]
print(array1d>5)
print(array3)
```

    [False False False False False  True  True  True  True]
    [6 7 8 9]
    


```python
array1=np.array([3,1,4,5,6,9])
print('원본 행렬:',array1)
sort_array1=np.sort(array1)
print('sorted된 행렬: ',sort_array1)
print('sorted 후 원본 행렬:',array1)
sort_array2=array1.sort()
print('sort()후 반환된 행렬:',sort_array2)
print('sort()후 원본 행렬:',array1)
```

    원본 행렬: [3 1 4 5 6 9]
    sorted된 행렬:  [1 3 4 5 6 9]
    sorted 후 원본 행렬: [3 1 4 5 6 9]
    sort()후 반환된 행렬: None
    sort()후 원본 행렬: [1 3 4 5 6 9]
    


```python
sort_array3=np.sort(array1)[::-1]
print(sort_array3)
```

    [9 6 5 4 3 1]
    


```python
array2d=np.array([[8,13],[7,1]])
print('원본 행렬:\n',array2d)
sort_array2d=np.sort(array2d,axis=0)
print('row 방향 정렬:\n',sort_array2d)
sort_array2d2=np.sort(array2d,axis=1)
print('column 방향 정렬:\n',sort_array2d2)
```

    원본 행렬:
     [[ 8 13]
     [ 7  1]]
    row 방향 정렬:
     [[ 7  1]
     [ 8 13]]
    column 방향 정렬:
     [[ 8 13]
     [ 1  7]]
    


```python
ordarray=np.arange(10)
nordarray=np.sort(ordarray)[::-1]
indexss=np.argsort(nordarray)
print(type(indexss))
print('행렬정렬시 인덱스',indexss)
```

    <class 'numpy.ndarray'>
    행렬정렬시 인덱스 [9 8 7 6 5 4 3 2 1 0]
    


```python
name_array=np.array(['Mark','Renjun','Jeno','Jaehyun','Jungwoo','Doyoung'])
score_array=np.array([67,89,23,45,90,100])
index_score=np.argsort(score_array)
print('성적 오름차순시 score의 index: ',index_score)
print('성적 오름차순시 name_array',name_array[index_score])
```

    성적 오름차순시 score의 index:  [2 3 0 1 4 5]
    성적 오름차순시 name_array ['Jeno' 'Jaehyun' 'Mark' 'Renjun' 'Jungwoo' 'Doyoung']
    


```python
A=np.array([[9,8,6],[10,5,2]])
B=np.array([[2,1],[7,9],[15,2]])
dot_product=np.dot(A,B)
print('행렬 내적 결괴:\n',dot_product)
```

    행렬 내적 결괴:
     [[164  93]
     [ 85  59]]
    


```python
transA=np.transpose(A)
print('A의 원본 행렬:\n',A)
print('A의 전치 행렬:\n',transA )
```

    A의 원본 행렬:
     [[ 9  8  6]
     [10  5  2]]
    A의 전치 행렬:
     [[ 9 10]
     [ 8  5]
     [ 6  2]]
    
