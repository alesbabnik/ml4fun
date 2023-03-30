points = {"!":[(1,2,4,3,4),(4,3,4,4,2),(3,1,4,3,6)],"?":[(6,5,8,4,3),(7,7,7,7,7),(8,6,8,8,9)]}
calc_point = (7,5,6,7,6)

def euclidean_distance(p: float, q: float) -> float:
    distance = 0
    for i in range(len(p)):
        distance += (p[i] - q[i]) ** 2
    return distance ** 0.5

class knn:
    def __init__(self, k:int=3) -> None:
        self.k = k

    def fit(self, points: dict) -> None:
        self.points = points

    def predict(self, new_point: tuple) -> str:
        distances = []

        for group in self.points:
            for point in self.points[group]:
                distance = euclidean_distance(point, new_point)
                distances.append([distance, group])

        votes = [i[1] for i in sorted(distances)[:self.k]]
        vote_result = 0
        for i in votes:
            if votes.count(i) > vote_result:
                vote_result = votes.count(i)
                result = i

        return result
    

knn = knn()
knn.fit(points)
print(knn.predict(calc_point))

        