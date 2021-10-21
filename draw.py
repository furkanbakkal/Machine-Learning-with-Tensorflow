# draw square in Python Turtle
from os import name
import turtle
import time 

def boxing(start_point_x, start_point_y,length,label):
    t = turtle.Turtle()
    turtle.title("Squid Game Challange")
    wn = turtle.Screen()
    wn.setup(1200, 800)
    wn.bgpic("test.gif")

    t.hideturtle()
    t.penup()                #don't d_raw when turtle moves
    t.goto(start_point_x-600, start_point_y-400)
    t.showturtle()           #make the turtle visible
    t.pendown()              

    # drawing first side
    t.forward(length) # Forward turtle by s units
    t.right(90) # Turn turtle by 90 degree

    # drawing second side
    t.forward(length) # Forward turtle by s units
    t.right(90) # Turn turtle by 90 degree

    # drawing third side
    t.forward(length) # Forward turtle by s units
    t.right(90) # Turn turtle by 90 degree

    # drawing fourth side
    t.forward(length) # Forward turtle by s units
    t.right(90) # Turn turtle by 90 degree

    t.write(label,font=("Verdana",10))

    time.sleep(3)


