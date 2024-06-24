function result = xi_func (m_iter, n_iter, eps1, eps2, r, a, beta_0)
    parfor i = 1:length(m_iter)
        result(i) = xi_func_serial(m_iter(i), n_iter(i), eps1, eps2, r, a, beta_0);
    end
end

function result = xi_func_serial (m, n, eps1, eps2, r, a, beta_0)
    x1 = -r/2;
    y1 = -r/2;
    x2 = r/2;
    y2 = -r/2;
    x3 = -r/2;
    y3 = r/2;
    function inside = isInsideTriangle(x, y)
        t1 = (x1-x)*(y2-y1) - (x2-x1)*(y1-y);
        t2 = (x2-x)*(y3-y2) - (x3-x2)*(y2-y);
        t3 = (x3-x)*(y1-y3) - (x1-x3)*(y3-y);
        inside = (t1 >= 0 & t2 >= 0 & t3 >= 0) | (t1 <= 0 & t2 <= 0 & t3 <= 0);
    end
    xi = 1 ./ (a.^2) .* integral2(@(x, y) (eps2.*isInsideTriangle(x,y)+eps1.*~isInsideTriangle(x,y)).*exp(1j.*(m.*beta_0.*x + n.*beta_0.*y)), -a/2, a/2, -a/2, a/2);
    result = xi;
end