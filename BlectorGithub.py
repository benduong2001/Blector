import bpy, bmesh
import math

expanse = 10
subunit = 100
domainList = [(-expanse + i*((2*expanse)/subunit)) for i in range(subunit)]

class Point:
    def placeholder(lst, index):
        if index >= len(lst):
            return None
        else:
            return lst[index]
    def __init__(self, coords):
        #placeholder = lambda lst, i: None if (i >= len(lst)) else lst[i]
        self.x = Point.placeholder(coords, 0)
        self.y = Point.placeholder(coords, 1)
        self.z = Point.placeholder(coords, 2)
        tempcoords = [self.x, self.y, self.z]
        self.coords = tuple([n for n in tempcoords if n != None])
        self.dimens = len([n for n in tempcoords if n != None])

class Vector:
    def pythag(hcoords, tcoords):
        # DONT DELETE; referenced in VectorGetNorm
        tempdists = [h-c for h, c in zip(hcoords, tcoords)]
        return (sum([i**2 for i in tempdists]))**(1/2)
    def placeholder(lst, index):
        if index >= len(lst):
            return None
        else:
            return lst[index]
    def VectorGetNorm(given_Vector):
        result = Vector.pythag(given_Vector.headc, given_Vector.tailc)
        return result
    def i():
        return Vector([1, 0, 0])
    def j():
        return Vector([0, 1, 0])
    def k():
        return Vector([0, 0, 1])
    def sbl():
        return [Vector.i(), Vector.j(), Vector.k()]
    def __init__(self, inpt):
        if type(inpt) == Point:
            self.headp = inpt.coords
            self.headc = inpt.coords
            self.zerovectlist = [0 for _ in range (inpt.dimens)]
            self.tailp = Point(self.zerovectlist)
            self.tailc = tuple(self.zerovectlist)
            
        elif type(inpt) in [list, tuple]: #is list or tuple
            
            if all((type(i) in [int, float]) for i in inpt): # 1 list
                self.headp = Point(inpt)
                self.headc = tuple(inpt) 
                self.zerovectlist = [0 for _ in range (len(inpt))]
                self.tailp = Point(self.zerovectlist)
                self.tailc = tuple(self.zerovectlist)
                
            elif all((type(i) == Point) for i in inpt): # 2 points
                # used by gradient!
                self.headp = inpt[0]
                self.headc = tuple(inpt[0].coords)
                
                self.tailp = inpt[1]
                self.tailc = tuple(inpt[1].coords)
            
        self.dimens = len(self.headc)
        
        self.length = self.magnitude = self.norm = Vector.VectorGetNorm(self)
        self.worldpoints = [self.tailp, self.headp]
    def __add__(self, other):
        assert type(other) == Vector
        otherVector = other
        if (self.dimens == otherVector.dimens):
            temp_add_list = [(s+o) for s, o in zip(self.headc, otherVector.headc)]
            return Vector(temp_add_list)
    def __mul__(self, other):
        if type(other) in [int, float]:
            scalar = other
            temp_scalar_list = [(s*scalar) for s in self.headc]
            return Vector(temp_scalar_list)
        elif type(other) == Vector:
            return self.dotprod(other)
    def dotprod(self, other):
        assert type(other) == Vector
        otherVector = other
        temp_dot =  sum([(s * o) for s, o in zip(self.headc, otherVector.headc)])
        return temp_dot

    def __subtract__(self, vector2):
        assert type(vector2) == Vector
        return self+(-1 * vector2)
    
    def angle(self, otherVector):
        """
        (q * r)/(||q|| * ||r||)
        """
        upper = (self.__mul__(otherVector))
        otherVectorNorm = (Vector.VectorGetNorm(otherVector))
        lower = self.norm * otherVectorNorm
        cosine = upper/lower
        return math.acos(cosine)
    def unit_vector(self):
        if (self.magnitude == 0):
            # PREVENT ZERO DIVISION ERROR
            return Vector(self.zerovectlist);
        else:
            base = (1/(self.magnitude))
            return Vector([(base*comp) for comp in self.headc])


class Rt:
    def __init__(self, basePoint, directionVector, asVector = False):
        self.basePoint = basePoint
        self.directionVector = directionVector
        self.worldpoints = []
        self.asVector = asVector
        """
        parameterization form: basepoint + T*directionvector
        """
    def BuildWorldslist(self):
        if self.asVector == True:
            self.worldpoints.append(self.basePoint)
            baseVector = Vector(self.basePoint)
            tvect = baseVector + (self.directionVector)
            tcoord = list(tvect.headc)
            tpoint = Point(tcoord)
            self.worldpoints.append(tpoint)
            return None
        baseVector = Vector(self.basePoint)
        for t in range(-10, 10):
            tvect = baseVector + ((self.directionVector).__mul__(t))
            tcoord = list(tvect.headc)
            tpoint = Point(tcoord)
            self.worldpoints.append(tpoint)
class Equation:
    def __init__(self, given_list):
        """IN STANDARD FORM, NOT 
            Y = MX + B form, so like
            1Y - MX = B form instead"""
        self.ca = given_list
        self.coeffs = given_list[:-1]
        self.answer = given_list[-1]
    def __mul__(self, scalar):
        return Equation([s*scalar for s in self.ca])
    def __add__(self, otherEquation):
        return Equation([a+b for a,b in zip(self.ca, otherEquation.ca)])
    def __sub__(self, otherEquation):
        return self.__add__(otherEquation.__mul__(-1))
    def __repr__(self):
        return str(self.ca)

class CoordFunc:
    def __init__(self, str_expr):
        self.str_expr = str_expr
        #self.fn = lambda n: eval(str_expr.replace("@", str(n)))        
    def fn(self, inpt):
        str_expr = self.str_expr
        if type(inpt) in [list, tuple]:  # inpt is a point P = [x, y]  
            if ("@" in str_expr):
                str_expr = str_expr.replace("@", str(inpt[0]))
            if ("!" in str_expr):
                str_expr = str_expr.replace("!", str(inpt[1]))
            if ("$" in str_expr):
                str_expr = str_expr.replace("$", str(inpt[2]))
        elif type(inpt) in [int, float]:  
            str_expr = str_expr.replace("@", str(inpt))
        try:
            answer = eval(str_expr)
        except:
            answer = None
        return answer
    def isPlaceholderCoordFunc(self):
        return self.str_expr == "0"

class vvf (Rt):
    def __init__(self, directionVector, basePoint=[0,0,0]):
        super(vvf, self).__init__(basePoint, directionVector)
    def BuildWorldslist(self):
        for t in domainList:
            tvect = Vector([dcf.fn(t) for dcf in self.directionVector])
            tcoord = list(tvect.headc)
            tpoint = Point(tcoord)
            self.worldpoints.append(tpoint)
            
               


class Gradient: #(vvf)
    # if basepoint is 2 components, add [0] to it, or fst?
    
    @staticmethod
    def fixgradpoint(basePoint, mvfObj):
        if (len(basePoint) ==3):
            return basePoint;
        elif (len(basePoint) == 2):
            x_num = basePoint[0]
            y_num = basePoint[1]
            z_num = mvfObj.apply(x_num, y_num)
            return [x_num, y_num, z_num] 
    
    def __init__(self, directionVector, mvfObj, basePoint = Point([0,0,0])):
        # [to uncomment this, uncom the pound in class header]
        ## super(Gradient,self).__init__(directionVector, Gradient.fixgradpoint(basePoint, mvfObj))
        # ensure point adjusted
        # ensure plugging in allows for x and y varb
        # 
        # ensure BuildWorldslist correct [no for loop]
        self.directionVector = directionVector;
        self.mvfObj = mvfObj;
        self.gradpoint(basePoint); # set basePoint
        self.setUnit(False);
    
    def setUnit(self, boolunit):
        self.unitForm = boolunit;
    
    def gradpoint(self, point):
        pointCoords = point.coords
        self.basePoint = Gradient.fixgradpoint(pointCoords, self.mvfObj);
    
    def BuildWorldslist(self):
        #baseVector = Vector(self.basePoint)
        self.worldpoints = []
        startPointCoords = self.basePoint
        directionPointCoords = [dcf.fn(self.basePoint) for dcf in self.directionVector]
        if (self.unitForm == True):
            unitScalar = (sum([(dpc*dpc) for dpc in directionPointCoords]))**(1/2)
            if (unitScalar == 0):
                # PREVENT ZERO DIVIDE ERROR
                directionPointCoords = [0*dpc for dpc in directionPointCoords]
            else:
                directionPointCoords = [(1/unitScalar)*dpc for dpc in directionPointCoords]
        startPoint = Point(startPointCoords)
        directionPoint = Point(directionPointCoords)
        endPointCoords = ([(d+b) for d, b in zip(directionPointCoords, startPointCoords)])
        endPoint = Point(endPointCoords)
        tvect = Vector([startPoint, endPoint])
        self.worldpoints.append(startPoint)
        self.worldpoints.append(endPoint)
        return tvect


class Plane:
    def __init__(self, first_inpt, secnd_inpt = None):
        equation_bool = (isinstance(first_inpt, Equation))
        normal_vector_point_pair_bool = ((isinstance(first_inpt, Vector)) and
                                         (isinstance(secnd_inpt, Point)))
        side_case_listeq_bool = ((type(first_inpt)==list) and (len(first_inpt)==4))
        if side_case_listeq_bool:
            
            Plane(Equation(first_inpt))
        elif equation_bool:
            # only one input, a list of 4 ints
            self.equation = first_inpt
            self.normal_vector = Vector(first_inpt.ca[:-1])
            self.coeffs = list(first_inpt.ca[:-1])
            self.anchor = (first_inpt.ca[-1])
        elif normal_vector_point_pair_bool:
            # 2 points; vector and point
            self.normal_vector = first_inpt;
            self.base_point = secnd_inpt;
            self.coeffs = list(first_inpt.headc)
            self.anchor = sum([c*v for c, v in zip(list(first_inpt.headc), 
                                                   list(secnd_inpt.coords))])
            self.equation = Equation(self.coeffs + [self.anchor])
        self.worldpoints = []
    def grounded(self):
        # assert (self.anchor == 0)
        self.anchor = 1;
        last_index = -1 #if unsure dimensions, or 2 if it is 3 dimn
        self.BuildWorldslist();
        for point_i in range(len(self.worldpoints)):
            point = self.worldpoints[point_i];
            edit_points_coords = list(point.coords)
            zcoeff = self.coeffs[last_index]
            if zcoeff == 0:
                edit_points_coords[last_index] -= 0
            else:
                edit_points_coords[last_index] -= (1/(zcoeff));
            point = Point(tuple(edit_points_coords))
            self.worldpoints[point_i] = point
    def BuildWorldslist(self):
        if (self.anchor == 0):
            self.grounded()
            self.anchor = 0;
            return;
        coefficients = {0: Vector.i(), 1: Vector.j(), 2: Vector.k()}
        intercept_pairs = []
        for coeff_i in range(len(self.coeffs)):
            intercept_pair = []
            axis_plane_coeffs = [coeff_j for coeff_j in range(len(self.coeffs))
                                 if coeff_i != coeff_j]
            for axis_plane_coeff in axis_plane_coeffs:
                temp_basis_vector = coefficients[axis_plane_coeff]
                coeff_value = self.coeffs[axis_plane_coeff]
                try:
                    carried = self.anchor * (1 / coeff_value)
                except ZeroDivisionError: 
                    carried = 0
                tempvector = temp_basis_vector * carried
                intercept_pair.append(tempvector)
            intercept_pairs.append(intercept_pair)
        worldpointstuple = list(set([ip.headc for pair in intercept_pairs for ip in pair]))
        
        self.worldpoints = [Point(p) for p in worldpointstuple]

class MVF: #MultiVariableFunction
    def __init__(self, multFuncString):
        self.DEFAULT_DOMAIN = domainList;
        self.X_STR_VARB = "@"
        self.Y_STR_VARB = "!"
        self.STR_VARBS =  [self.X_STR_VARB, self.Y_STR_VARB]
        self.multFuncString = multFuncString;
        self.worldterrain = [];
    
    def apply(self, x_num, y_num):
        #fstr = "".join(list(self.multFuncString)[:])
        fstr = self.multFuncString
        fstr = fstr.replace("@", str(x_num))
        fstr = fstr.replace("!", str(y_num))
        return eval(fstr)
        
    
    def BuildWorldslist(self):
        def traceX(xConst, fstr, nonlocalList):
            fstr = fstr.replace("@", str(xConst))
            worldtrace = []
            for yConst in self.DEFAULT_DOMAIN:
                tempfstr = fstr
                tempfstr = tempfstr.replace("!", str(yConst))
                try:
                    output = eval(tempfstr)
                    pt = Point([xConst, yConst, output])
                    worldtrace.append(pt)
                except:
                    pt = Point([None, None, None])
                    worldtrace.append(pt)
            nonlocalList.append(worldtrace)
            return;
        for x_i in self.DEFAULT_DOMAIN:
            traceX(x_i, self.multFuncString, self.worldterrain)      
                        

class Blector:
    PLANE_NUM = 0
    PATH_NUM = 0
    TERRAIN_NUM = 0
    FIELD_NUM = 0
    def setEdge(self, bm, vertA, vertB):
        if ((vertA != None) and (vertB != None)):
            newEdges = bm.edges.new((vertA, vertB))
            return newEdges;
        else:
            return None
    def setVert(self, bm, pointA):
        outsideDomain = tuple([]) #[None, None, None]
        if ((pointA.coords) != outsideDomain):
            newVert = bm.verts.new((pointA.coords))
            return newVert;
        else:
            return None;
            
            
    
    def __init__(self):
        self.object_list = []
    def addPath(self, mathobj):
        # addPath: vvf, rt
        # accessible from objectmode
        # bmesh
        # ...not visible entirely

        mesh = bpy.data.meshes.new("mesh")  # add a new mesh
        obj = bpy.data.objects.new("PATH" + str(Blector.PATH_NUM), mesh)  # add a new object using the mesh
        Blector.PATH_NUM += 1
        scene = bpy.context.scene
        scene.objects.link(obj)  # put the object into the scene (link)
        scene.objects.active = obj  # set as the active object in the scene
        obj.select = True  # select object #"""

        ############# start of construction #########
        temp_obj_vertices = []
        me = bpy.context.object.data
        bm = bmesh.new() #"""
        bm.from_mesh(me)

        for point in mathobj.worldpoints:
            temp_obj_vertice = bm.verts.new(point.coords)
            temp_obj_vertices.append(temp_obj_vertice)


        for i in range(len(temp_obj_vertices) - 1):
            tempVertA = temp_obj_vertices[i]
            tempVertB = temp_obj_vertices[i + 1]
            bm.edges.new((tempVertA, tempVertB))

        #bm.faces.new(tuple(temp_obj_vertices))
        # make the bmesh the object's mesh
        bm.to_mesh(me)  
        bm.free()        
        #bmesh.update_edit_mesh(obj.data)
    def addPlane(self, mathobj):
        assert type(mathobj) == Plane
        mesh = bpy.data.meshes.new("mesh")  # add a new mesh
        obj = bpy.data.objects.new("PLANE" + str(Blector.PLANE_NUM), mesh)  # add a new object using the mesh
        Blector.PLANE_NUM += 1
        scene = bpy.context.scene
        scene.objects.link(obj)  # put the object into the scene (link)
        scene.objects.active = obj  # set as the active object in the scene
        obj.select = True  # select object #"""

        ############# start of construction #########
        temp_obj_vertices = []
        me = bpy.context.object.data
        bm = bmesh.new() #"""
        bm.from_mesh(me)

        for point in mathobj.worldpoints:
            temp_obj_vertice = bm.verts.new(point.coords)
            temp_obj_vertices.append(temp_obj_vertice)


        for i in range(len(temp_obj_vertices) - 1):
            tempVertA = temp_obj_vertices[i]
            tempVertB = temp_obj_vertices[i + 1]
            bm.edges.new((tempVertA, tempVertB))
        #ax + by + cz = d 
        #if there is 2 or more zero in coefficients
        #(one zero is allowed)
        #or if the anchor (d) is zero
        #then the function must accomodate these sidecases

        zeroCoeffsBool = (((mathobj.coeffs).count(0)) > 1)
        zeroAnchorBool = (mathobj.anchor == 0)
        if ((zeroCoeffsBool==False)): # and (zeroAnchorBool==False)):
            # assuming none ofthe bools are true
            finalvert = temp_obj_vertices[-1]
            firstvert = temp_obj_vertices[0]
            bm.edges.new((finalvert, firstvert)) #close the loop
            bm.faces.new(tuple(temp_obj_vertices))

            bm.to_mesh(me)
            bm.free()
            return None;
        bm.to_mesh(me)  
        bm.free()
        return None;
        #bmesh.update_edit_mesh(obj.data)

   
    def addTerrain(self, mathobj):
        mesh = bpy.data.meshes.new("mesh")  # add a new mesh
        obj = bpy.data.objects.new("TERRAIN" + str(Blector.TERRAIN_NUM), mesh)  # add a new object using the mesh
        Blector.TERRAIN_NUM += 1
        scene = bpy.context.scene
        scene.objects.link(obj)  # put the object into the scene (link)
        scene.objects.active = obj  # set as the active object in the scene
        obj.select = True  # select object #"""
        
        me = bpy.context.object.data
        bm = bmesh.new() #"""
        bm.from_mesh(me)
        
        temp_obj_traces = []
        temp_obj_vertices = []
        
        len_row = len(mathobj.worldterrain[0])
        
        for trace_i in range(len(mathobj.worldterrain )):
            trace = mathobj.worldterrain[trace_i]
            for point_i in range(len(trace)):
                point = trace[point_i]
                temp_obj_vertice = self.setVert(bm, point)
                temp_obj_vertices.append(temp_obj_vertice)
            
            
        for vert_i in range(len(temp_obj_vertices) - 1):
            lastrowbool = vert_i%len_row == (len_row - 1)
            lastitembool = vert_i//len(mathobj.worldterrain)  == (len(mathobj.worldterrain) - 1)
            if ((not lastrowbool) and (not lastitembool)):
                vertA = temp_obj_vertices[vert_i]
                vertB = temp_obj_vertices[vert_i+1]
                vertC = temp_obj_vertices[vert_i + len_row]
                vertD = temp_obj_vertices[vert_i + 1 + len_row]
                self.setEdge(bm, vertA, vertB)
                self.setEdge(bm, vertA, vertC)
                #
                #bm.faces.new((vertA, vertB, vertC, vertD))

        bm.to_mesh(me)  
        bm.free()
    def addField(self, mathobj):
        mesh = bpy.data.meshes.new("mesh")  # add a new mesh
        obj = bpy.data.objects.new("FIELD" + str(Blector.FIELD_NUM), mesh)  # add a new object using the mesh
        Blector.FIELD_NUM += 1
        scene = bpy.context.scene
        scene.objects.link(obj)  # put the object into the scene (link)
        scene.objects.active = obj  # set as the active object in the scene
        obj.select = True  # select object #"""
        
        me = bpy.context.object.data
        bm = bmesh.new() #"""
        bm.from_mesh(me)
        
        for vector_segment in mathobj.worldfield:
            pointA, pointB = vector_segment
            tempVertA = bm.verts.new(pointA.coords)
            tempVertB = bm.verts.new(pointB.coords)
            bm.edges.new((tempVertA, tempVertB))
        
        bm.to_mesh(me)  
        bm.free()  
    def addObject(self, mathobj):
        if type(mathobj) in [Rt, vvf, Gradient]:
            self.addPath(mathobj)
        elif type(mathobj) in [Plane]:
            self.addPlane(mathobj)
        elif type(mathobj) in [MVF]:
            self.addTerrain(mathobj);
        elif type(mathobj) in [Vector_Field]:
            self.addField(mathobj)
      

Blect = Blector()


def make(mathObj):
    mathObj.BuildWorldslist()
    Blect.addObject(mathObj)

def gradField(gradObj):
    k = 7
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            tempGrad = Gradient(gradObj.directionVector, gradObj.mvfObj)
            tempGrad.gradpoint(Point([i, j, 0]))
            tempGrad.setUnit(True)
            make(tempGrad)

class Field:
    def __init__(self):
        pass
        
class Vector_Field (Field):
    def __init__(self, coord_func_vector, basePoint=[0,0,0]):
        assert type(coord_func_vector) == vvf;
        assert type(coord_func_vector.directionVector[0]) == CoordFunc;
        self.coord_func_vector = coord_func_vector;
        self.worldfield = []
        self.radius = 5
        self.setUnit(True);
    
    def setUnit(self, boolunit):
        self.unitForm = boolunit;
    
    def _pad_2d_points(self, point):
        if type(point) in [tuple, list]:
            if len(point) == 2:
                return point + [0]
        if type(point) in [Point]:
            if len(point.coords) == 2:
                return Point(list(point.coords) + [0])
        return point
    def _addVectorSegment(self, input_point_list):
        
        t_point_list = [cf.fn(input_point_list) for cf in self.coord_func_vector.directionVector]
        if None in t_point_list:
            # if some math error (like outside of domain, div by zero occurs, abandon this point
            return None
        input_point_list = self._pad_2d_points(input_point_list)
        t_point_list = self._pad_2d_points(t_point_list)
                
        directionPointCoords = t_point_list
        if (self.unitForm == True):
            unitScalar = (sum([(dpc*dpc) for dpc in directionPointCoords]))**(1/2)
            if (unitScalar == 0):
                # PREVENT ZERO DIVIDE ERROR
                directionPointCoords = [0*dpc for dpc in directionPointCoords]
            else:
                directionPointCoords = [(1/unitScalar)*dpc for dpc in directionPointCoords]
        t_point_list = directionPointCoords        
        input_vect = Vector(input_point_list)
        input_point = Point(input_point_list)        
        t_vect = Vector(t_point_list)
        end_vect = input_vect + t_vect
        end_point = Point(end_vect.headc)
        end_point_list = list(end_vect.headc)
        vector_segment = [input_point, end_point]
        self.worldfield.append(vector_segment)
    def _BuildVectorField3D(self):
        r = self.radius
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                for k in range(-r, r+1):
                    input_point_list = [i, j, k]
                    self._addVectorSegment(input_point_list)
    def _BuildVectorField2D(self):
        r = self.radius
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                input_point_list = [i, j]
                self._addVectorSegment(input_point_list)
    def BuildWorldslist(self):
        if len(self.coord_func_vector.directionVector) == 2:
            self._BuildVectorField2D() # 2D
        elif len(self.coord_func_vector.directionVector) == 3:
            # if the third coordfunc isn't just a placeholder 
            if self.coord_func_vector.directionVector[2].isPlaceholderCoordFunc():
                self._BuildVectorField2D()
            else:
                self._BuildVectorField3D()
