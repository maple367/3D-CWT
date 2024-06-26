import decimal, math
from decimal import Decimal
from . import errors, constants
decimal.getcontext().prec = 50

"""The number class
Complex number Decimal class to keep precision while supporting complex numbers
"""

    
class ComplexDecimal(object):
    def __init__(self,real,i="0.0"):
        if type(real) != Decimal: self.real = Decimal(real)
        else: self.real = real
        if type(i) != Decimal: self.imaginary = Decimal(i)
        else: self.imaginary = i

        self.e = constants.e
        self.pi = constants.pi
        
    #Basic mathamtical functions ----------------------------------------
    def conj(self):
        """Returns the conjugate for a number"""
        return ComplexDecimal(self.real,-self.imaginary)
        
    def re(self):
        """Returns real portion"""
        return self.real
        
    def im(self):
        """Returns imaginary portion"""
        return self.imaginary
        
    def floor(self):
        """Floors a number"""
        return ComplexDecimal(self.real.to_integral(), self.imaginary.to_integral())
        
    def ceil(self):
        """Ceilings a number"""
        return ComplexDecimal(math.ceil(self.real),math.ceil(self.imaginary))
        
    def round(self,prec=0):
        """Rounds a number to prec decimal places"""
        return ComplexDecimal(round(self.real,prec),round(self.imaginary,prec))
        
    def exp(self):
        """Returns e^self"""
        if self.imaginary == 0:
            return ComplexDecimal(str(self.e ** self.real ))
        return ComplexDecimal(self.e**(self.real) * Decimal(self.cos_raw(self.imaginary)), self.e**(self.real) * Decimal(self.sin_raw(self.imaginary)))
        
    def isPrime_raw(self,x):
        if x<=1: return False
        for i in range(2,int(math.floor(math.sqrt(x))) + 1):
            if x%i==0: return False
        return True
        
    def isPrime(self):
        """Gets if self is prime"""
        if self.imaginary == 0: 
            return self.isPrime_raw(self.real)
        if self.real == 0:
            return self.isPrime_raw(abs(self.imaginary))
        return self.isPrime_raw(self.real * self.real + self.imaginary*self.imaginary)
    
    def phase(self):
        """Returns angle between x axis"""
        return Decimal(math.atan2(self.imaginary , self.real))
        
    def polar(self): #Converts to polar
        return (abs(self), self.phase())
        
    def toRect(self,r,phi):
        return ComplexDecimal(r) * ComplexDecimal(self.cos_raw(phi) , self.sin_raw(phi))
        
        
    #Trig --------------------------------------------
    #1/functions, log, 
    
    def cos_raw(self,x,prec=Decimal("0.0000000000000000000001")):
        """Raw cosine function for real numbers only"""
        p = Decimal("0")
        s = t = Decimal("1.0")
        while abs(t/s) > prec:
            p+=Decimal("1")
            t = (-t * x * x) / ((Decimal("2") * p - Decimal("1")) * (Decimal("2") * p))
            s += t
        if round(s,50) in [1,0,-1]: #If it's close enough round to get perfect answers (No one likes 1E-70 as cos(pi))
            return Decimal(round(s,50))
        return s
    
    def sin_raw(self,x,prec=Decimal("0.00000000000000000000001")):
        return self.cos_raw(x+self.pi/Decimal("2"))
        
    def tan_raw(self,x,prec=Decimal("0.00000000000000000000001")):
        return self.sin_raw(x,prec) / self.cos_raw(x,prec)
        
    def atan_raw(self,x): #Not relatively accurate
        returned = Decimal("0")
        for j in range(0,50):
            i = Decimal(j)
            a = Decimal("-1")**i * x ** (Decimal("2")*i + Decimal("1"))
            b = Decimal("2")*i + Decimal("1")
            returned += a/b
        return returned
    
    def cos(self):
        """Compute cos of self"""
        if self.imaginary == 0:
            return ComplexDecimal(self.cos_raw(self.real))
        try:
            returned = ComplexDecimal(
                self.cos_raw(self.real) * ComplexDecimal(self.imaginary).cosh().real,
                -self.sin_raw(self.real) * ComplexDecimal(self.imaginary).sinh().real
            )
            returned.imaginary = -returned.imaginary
            return returned
        except: return ComplexDecimal("inf")
        
    def sin(self):
        """Compute sin of self"""
        if self.imaginary == 0:
            return ComplexDecimal(self.sin_raw(self.real))
        try: 
            returned = ComplexDecimal(
                self.sin_raw(self.real) * ComplexDecimal(self.imaginary).cosh().real,
                -self.cos_raw(self.real) * ComplexDecimal(self.imaginary).sinh().real
            )
            returned.imaginary = -returned.imaginary
            return returned
        except: return ComplexDecimal("inf")
        
    def tan(self):
        """Compute tangent of self"""
        try: return self.sin() / self.cos()
        except: return ComplexDecimal("inf")
        
    def acos(self):
        """Attempt to compute arcosine"""
        if self.imaginary == 0:
            return ComplexDecimal(math.acos(self))
        A = ComplexDecimal(((Decimal(1) + self.real)**Decimal(2) + self.imaginary**Decimal(2))**Decimal(0.5) - ((Decimal(1) - self.real)**Decimal(0.5) + self.imaginary**Decimal(2))**Decimal(0.5)) / ComplexDecimal(2)
        B = ComplexDecimal(((Decimal(1) + self.real)**Decimal(2) + self.imaginary**Decimal(2))**Decimal(0.5) + ((Decimal(1) - self.real)**Decimal(0.5) + self.imaginary**Decimal(2))**Decimal(0.5)) / ComplexDecimal(2)
        return ComplexDecimal(A.acos(), -(B+(B*B - ComplexDecimal(1))**ComplexDecimal(0.5)).ln() )
        
    def asin(self):
        """Attempt to compute arcsine"""
        if self.imaginary == 0:
            return ComplexDecimal(math.asin(self))
        A = ComplexDecimal(((Decimal(1) + self.real)**Decimal(2) + self.imaginary**Decimal(2))**Decimal(0.5) - ((Decimal(1) - self.real)**Decimal(0.5) + self.imaginary**Decimal(2))**Decimal(0.5)) / ComplexDecimal(2)
        B = ComplexDecimal(((Decimal(1) + self.real)**Decimal(2) + self.imaginary**Decimal(2))**Decimal(0.5) + ((Decimal(1) - self.real)**Decimal(0.5) + self.imaginary**Decimal(2))**Decimal(0.5)) / ComplexDecimal(2)
        return ComplexDecimal(A.asin(), (B+(B*B - ComplexDecimal(1))**ComplexDecimal(0.5)).ln() )
    
    def atan(self):
        """Attempt to compute arctangent"""
        result = (ComplexDecimal(0,1)+self) / (ComplexDecimal(0,1) - self)
        return ComplexDecimal(0,0.5) * result.ln()
        
        
    def sinh(self):
        """Hyperbolic sine of self"""
        returned = (self.exp() - (-self).exp()) / ComplexDecimal("2")
        returned.imaginary = -returned.imaginary
        return returned
        
    def cosh(self):
        """Hyperbolic cosine of self"""
        returned = (self.exp() + (-self).exp()) / ComplexDecimal("2")
        returned.imaginary = -returned.imaginary
        return returned
        
    def tanh(self):
        """Hyperbolic tangent of self"""
        return self.sinh()/self.cosh()
        
    def acosh(self):
        """Arc hyperbolic cosine"""
        returned = self + (self*self - ComplexDecimal(1))**0.5
        return returned.ln()
        
    def asinh(self):
        """Arc hyperbolic sine"""
        returned = self + (self*self + ComplexDecimal(1))**0.5
        return returned.ln()
        
    def atanh(self):
        """Arc hyperbolic  tan"""
        a = (ComplexDecimal(1)+self).ln() - (ComplexDecimal(1)-self).ln()
        return ComplexDecimal(0.5) * a
        
    def ln_raw(self,x):
        returned = Decimal("0")
        for i in range(0,25):
            a = Decimal("1") / (Decimal(i) * Decimal("2") + Decimal("1"))
            b = ( (x - Decimal("1"))/(x + Decimal("1")) ) ** (Decimal(i) * Decimal("2") + Decimal("1"))
            returned += a*b
        return returned*2
    
    def ln(self):
        """Natural logarithim of self"""
        if self.imaginary == 0:
            try: return ComplexDecimal(self.ln_raw(self.real))
            except: pass
        p = self.polar()
        return ComplexDecimal(self.ln_raw(p[0]), p[1])
    
    def log(self,x=None):
        """Compute log of number in base x"""
        if not x: x = self.e
        if self.imaginary == 0:
            try: return ComplexDecimal(math.log(self.real,x))
            except: pass
        p = self.polar()
        return ComplexDecimal(math.log(p[0],x), p[1])
        
    def log10(self):
        """Compute log base 10"""
        if self.imaginary == 0:
            return ComplexDecimal(math.log10(self.real))
        return self.ln() / ComplexDecimal("10").ln()
        
    #Replace builtins --------------------------------
    # power, comparasion,
    #To float, int, complex
    
    def __str__(self):
        """Converts to str"""
        if self.imaginary == 0:
            return str(self.real)
        return (str(self.real)+"+"+str(self.imaginary)+"i").replace("+-","-")
        
    def __abs__(self):
        """Absolute value"""
        return (self.real**2 + self.imaginary**2).sqrt()
    
    def __add__(self,other):
        """Add 2 numbers together"""
        return ComplexDecimal(self.real+other.real, self.imaginary+other.imaginary)
    
    def __sub__(self,other):
        """Subtract 2 numbers"""
        return ComplexDecimal(self.real-other.real, self.imaginary-other.imaginary)
    
    def __mul__(self,other):
        """Multiply 2 numbers"""
        if self.imaginary == 0 and other.imaginary == 0:
            return ComplexDecimal(self.real*other.real,"0")
        return ComplexDecimal( self.real*other.real - self.imaginary*other.imaginary,self.real*other.imaginary + self.imaginary*other.real )
   
    def __div__(self,other):
        """Divide 2 numbers"""
        if self.imaginary == 0 and other.imaginary == 0:
            return ComplexDecimal(self.real/other.real,"0")
        a = (self.real*other.real + self.imaginary*other.imaginary)/(other.real*other.real + other.imaginary*other.imaginary)
        b = (self.imaginary*other.real - self.real*other.imaginary)/(other.real*other.real + other.imaginary*other.imaginary)
        return ComplexDecimal(a,b)
    
    def __truediv__(self,other):
        """Divide 2 numbers"""
        if self.imaginary == 0 and other.imaginary == 0:
            return ComplexDecimal(self.real/other.real,"0")
        a = (self.real*other.real + self.imaginary*other.imaginary)/(other.real*other.real + other.imaginary*other.imaginary)
        b = (self.imaginary*other.real - self.real*other.imaginary)/(other.real*other.real + other.imaginary*other.imaginary)
        return ComplexDecimal(a,b)
    
    def __neg__(self):
        """Negates a number"""
        return ComplexDecimal(-self.real,-self.imaginary)
    
    def __pos__(self):
        """Positive number"""
        return ComplexDecimal(abs(self.real),abs(self.imaginary))
    
    def __inverse__(self):
        """1/ number"""
        return ComplexDecimal(1)/self
    
    def __mod__(self,other):#a%b = a + b * ciel(-a/b) 
        """Modolus"""
        return self+ other* ((-self/other).ceil())
        
    #Powers
    def __pow__(self,other):
        """Powers :D"""
        if self.imaginary == 0 and other.imaginary == 0 and self.real >= 0:
            return ComplexDecimal(self.real**other.real)
        if other.imaginary == 0:
            polar = self.polar()
            return self.toRect(polar[0]**other.real, polar[1]*other.real)
        elif other.real == 0:
            a = ComplexDecimal(self.real); b = ComplexDecimal(self.imaginary)
            c = ComplexDecimal(other.real); d = ComplexDecimal(other.imaginary)
            x = (-d * (b/a).atan()).exp() * ((a*a+b*b).ln() * d / ComplexDecimal(2)).cos() 
            y = ComplexDecimal(0,1) * (-d * (b/a).atan()).exp() * ((a*a+b*b).ln() * d / ComplexDecimal(2)).sin()
            return x+y

        b = other.real; c = other.imaginary
        returned = self**(ComplexDecimal(b)) * self**(ComplexDecimal(0,1) * ComplexDecimal(c))
        returned.imaginary = -returned.imaginary
        return returned
    
    #Additional conversions
    def __complex__(self): 
        """Convert to complex"""
        return complex(float(self.real),float(self.imaginary))
    def __int__(self):
        """Convert to int"""
        return int(self.real)
    def __float__(self):
        """Convert to float"""
        return float(self.real)
        
    #Comparasions
    def __lt__(self,other): #>
        if self.imaginary != 0 or other.imaginary != 0:
            raise errors.ComparasionError("Complex comparasion is not supported")
        return self.real < other.real
        
    def __le__(self,other): #>=
        if self.imaginary != 0 or other.imaginary != 0:
            raise errors.ComparasionError("Complex comparasion is not supported")
        return self.real <= other.real
        
    def __eq__(self,other): #==
        if self.real == other.real and self.imaginary == other.imaginary:
            return True
        return False
        
    def __ne__(self,other): #!=
        return not self.__eq__(other)
        
    def __gt__(self,other): #<
        if self.imaginary != 0 or other.imaginary != 0:
            raise errors.ComparasionError("Complex comparasion is not supported")
        return self.real > other.real
        
    def __ge__(self,other): #<=
        if self.imaginary != 0 or other.imaginary != 0:
            raise errors.ComparasionError("Complex comparasion is not supported")
        return self.real >= other.real
        
    #Some things reimplemented from decimal class for complex ComplexDecimals
    #================================================================
    def copy_abs(self): return abs(self)
    def copy_negate(self): return -self
    
    def copy_sign(self,other):
        if other < 0:
            return -abs(self)
        return abs(self)
        
    def is_finite(self):
        return self.real.is_finite() and self.imaginary.is_finite()
    
    def is_infinite(self):
        return not self.is_finite()
    
    def is_nan(self):
        return self.real.is_nan() or self.imaginary.is_nan()
        
    def is_signed(self):
        return self < ComplexDecimal(0)
    
    def is_zero(self):
        return self.real.is_zero() and self.imaginary.is_zero()
    
    def radix(self): return Decimal(10)
    def sqrt(self): 
        if self.imaginary.is_zero() and self.real >= 0:
            return ComplexDecimal(self.real.sqrt())
        return self ** ComplexDecimal("0.5")
        
    def to_eng_string(self): 
        returned = self.real.to_eng_string() + " + " + self.imaginary.to_eng_string + "i"
        return returned.replace(" + -"," - ")
    
    
    
    
