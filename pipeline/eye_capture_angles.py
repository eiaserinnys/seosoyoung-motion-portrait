"""Invert the runtime's fixed requested-to-achieved angle mapping."""

def achieved(c,yaw,pitch):
    y=yaw/45;p=pitch/30
    return [c[0]*y+c[1]*y*p,c[2]*p+c[3]*y*y+c[4]*p*p]

def inverse_achieved(c,yaw,pitch):
    p=pitch/c[2]
    for _ in range(30):
        denominator=c[0]+c[1]*p
        y=yaw/denominator
        error=c[2]*p+c[3]*y*y+c[4]*p*p-pitch
        derivative=c[2]+2*c[4]*p-2*c[3]*c[1]*y*y/denominator
        step=error/derivative;p-=step
        if abs(step)<1e-14:break
    result=[45*yaw/(c[0]+c[1]*p),30*p]
    assert max(abs(a-b) for a,b in zip(achieved(c,*result),[yaw,pitch]))<1e-9
    return result
