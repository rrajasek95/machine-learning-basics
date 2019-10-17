import pymunk

class Engine():
    def __init__(self):
        body = pymunk.Body(1, 1)
        body.position = 0, 0
        
        poly = pymunk.Poly.create_box(body)
        space = pymunk.Space()
        space.gravity = 0, -10
        space.add(body, poly)
        
        self.__body = body
        self.__space = space
        

    def queryPosition(self):
        position = self.__body.position
        self.__space.step(1)
        return position
        
        
        