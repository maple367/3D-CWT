function inside = isInsideTriangle(x, y, r)
    x1 = -r/2-x;
    y1 = -r/2-y;
    x2 = r/2-x;
    y2 = -r/2-y;
    x3 = -r/2-x;
    y3 = r/2-y;
    t1 = (x1)*(y2) - (x2)*(y1);
    t2 = (x2)*(y3) - (x3)*(y2);
    t3 = (x3)*(y1) - (x1)*(y3);
    inside = (t1 >= 0 & t2 >= 0 & t3 >= 0) | (t1 <= 0 & t2 <= 0 & t3 <= 0);
end