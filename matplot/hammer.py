import matplotlib
@mlab.animate(delay=10)
def anim():
    f = mlab.gcf()
    x=1
    while 1:
        f.scene.camera.azimuth(.5)
        f.scene.render()
        mlab.savefig("/home/christopher/code/Physics/image{}.png".format(x), figure=f, size=(1920,1080))
        x += 1
        yield

a = anim() # Starts the animation.
