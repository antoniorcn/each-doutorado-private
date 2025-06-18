import cv2
import math
import geometry_shapes as geom


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def integer(self):
        return int(self.x), int(self.y)

class Polygon(object):
    def __init__(self, shape_id):
        self.id = shape_id


class Rectangle(Polygon):
    def __init__(self, shape_id, shape):
        super(self.__class__, self).__init__(shape_id)
        self.shape = shape


class Square(Rectangle):
    pass


class PartType(object):
    def __init__(self, nome, color, thik):
        self.nome = nome
        self.color = color
        self.thickness = thik
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.radius = 0

    def get_circle_colision(self, other_circle):
        dx = abs(self.x - other_circle.x)
        dy = abs(self.y - other_circle.y)
        h = math.sqrt(dx * dx + dy * dy)
        rad_distance = self.radius + other_circle.radius
        return h/rad_distance

    def get_box_colision(self, other_rect):
        if ((self.x < (other_rect.x + other_rect.w) and (self.x + self.w) > other_rect.x) and
                (self.y < (other_rect.y + other_rect.h) and (self.y + self.h) > other_rect.y)):
            return 1
        else:
            return 0

    def get_box_circle_colision(self, other_box):
        return geom.circle_rectangle_collide(self, other_box)

    def get_part_info(self):
        info = {"top_left": (self.x, self.y),
                "top_right": (self.x + self.w, self.y),
                "bottom_left": (self.x, self.y + self.h),
                "bottom_right": (self.x + self.w, self.y + self.h),
                "center": (int(self.x + self.w/2), int(self.y + self.h/2)),
                "width": self.w,
                "height": self.h}
        return info

    def __str__(self):
        text = "x(" + str(self.x) + ") y(" + str(self.y) + ") w(" + str(self.w) + ") h(" + str(self.h) + ")"
        return text


class CirclePart(PartType):
    def __init__(self, nome, color, thik, x, y, radius):
        super(self.__class__, self).__init__(nome=nome, color=color, thik=thik)
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, mat):
        cv2.circle(mat,
                   center=(self.x, self.y),
                   radius=int(self.radius),
                   color=self.color, thickness=self.thickness)


class EllipsePart(PartType):
    def __init__(self, nome, color, thik, x, y, w, h):
        """
        :param nome: string
        :param color: tuple
        :param thik: int
        :param x: int
        :param y: int
        :param w: int
        :param h: int
        """
        super(self.__class__, self).__init__(nome=nome, color=color, thik=thik)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_middle(self):
        middle_x = self.x + (self.w / 2.0)
        middle_y = self.y + (self.h / 2.0)
        return Point(middle_x, middle_y)

    def get_point(self):
        return self.x, self.y

    def get_end_point(self):
        return self.x + self.w, self.y + self.h

    def get_size(self):
        return self.w, self.h

    def draw(self, mat):
        cv2.rectangle(mat,
                      pt1=(self.x, self.y),
                      pt2=(self.x + self.w, self.y + self.h),
                      color=self.color, thickness=self.thickness)


class BoxPart(PartType):
    def __init__(self, nome, color, thik, x, y, w, h):
        super(self.__class__, self).__init__(nome, color, thik)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def draw(self, mat):
        cv2.rectangle(mat,
                      pt1=(self.x, self.y),
                      pt2=(self.x + self.w, self.y + self.h),
                      color=self.color, thickness=self.thickness)
