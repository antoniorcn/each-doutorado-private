import math


def clamp_on_range(x, minimo, maximo):
    if x < minimo:
        return minimo
    elif maximo < x:
        return maximo
    else:
        return x


def clamp_on_rectangle(c, r):
    clamp = {"x": clamp_on_range(c.x, r.x, r.x + r.w), "y": clamp_on_range(c.y, r.y, r.y + r.h)}
    return clamp


def circle_point_collide(c, ponto):
    dx = abs(c.x - ponto["x"])
    dy = abs(c.y - ponto["y"])
    h = math.sqrt(dx ** 2 + dy ** 2)
    return h < c.radius


def circle_rectangle_collide(c, r):
    clamped = clamp_on_rectangle(c, r)
    return circle_point_collide(c, clamped)
