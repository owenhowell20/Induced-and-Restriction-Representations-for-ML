### compute the restriction of SO(3) representation to an SO(2) representation
def compute_restriction_SO3( rep_dict ):
    rep_out = {}
    for k in rep_dict.keys():

        ### if non-zero, contains one copy of each SO(2) irr
        if ( rep_dict[k] != 0 ):
            
            multplicity = int( rep_dict[k] )
            for k_s in range(0, int(k) + 1 ):
 
                try:
                    rep_out[ k_s ] = rep_out[ k_s ] + multplicity
                except:
                    rep_out[ k_s ] = multplicity
            
    return rep_out
        
### compute the dimension of a so3 representation
### input is dict form
def compute_SO3_dimension( rep_dict ):

    d_so3 = 0
    for k in rep_dict.keys():

        d_so3 = d_so3 + (2 * int(k) + 1) * int( rep_dict[k] )


    return d_so3


### compute the dimension of a so2 representation
### input is dict form
def compute_SO2_dimension( rep_dict_input  ):
    dval = 0
    for k in rep_dict_input.keys():
        rho_number = rep_dict_input[k]

        if(int(k)==0):
            dval = dval + rho_number
        if (int(k)!=0):
            dval = dval + 2*rho_number

    return dval


### return the tensor product of two so3 irriducible reps
### return dict
def compute_tensor_SO3( l1 , l2 ):
    l1 = int(l1)
    l2 = int(l2)
    rep_out = {}

    for i in range(  abs( l1 - l2 ) , l1 + l2  + 1  ):
        rep_out[i] = 1

    return rep_out


### return the tensor product of two so2 irriducible reps
def compute_tensor_SO2( k1 , k2 ):
    k1 = int(k1)
    k2 = int(k2)
    rep_out = {}

    if (k1==0 and k2==0):
        rep_out[0] = 1

    if (k1==0 and k2!=0):
        rep_out[k2] = 1

    if (k1!=0 and k2==0):
        rep_out[k1] = 1

    if (k1==k2 and k1!=0 and k2!=0):
        rep_out[int(0)] = 2
        rep_out[int(k1)+int(k1)] = 1

    if (k1!=k2 and k1!=0 and k2!=0):
        rep_out[ abs( int(k1)-int(k2) ) ] = 1
        rep_out[ int(k1)+int(k2) ] = 1


    return rep_out


### input is dict of SO3 rep and l integer, output is dict of SO3 rep
### compute the tensor product of SO3 rep and l irriducible
def compute_tensor_SO3_l_fold( rep_dict , l ):
    if l==0:
        return rep_dict

    rep_out = {}

    for k2 in rep_dict.keys():
                
        multplicity = int( rep_dict[ k2 ] )

        ### if non-zero, contains one copy of each SO(2) irr
        if ( multplicity != 0 ):
            
            tp = compute_tensor_SO3( l , k2 )
           
            for out in tp.keys():
                
                try:
                    rep_out[out] = rep_out[out] + int(multplicity)*int(tp[out])
      
                except:
                    
                    rep_out[out] = int(multplicity)*int(tp[out])
    
    return rep_out



### input is dict of SO2 rep and l integer, output is dict of SO2 rep
### compute the tensor product of SO2 rep and l restriction so3 rep
def compute_tensor_SO2_l_fold( rep_dict , l ):
    if l==0:
        return rep_dict

    rep_out = {}
    for k1 in range(0,l+1):
        for k2 in rep_dict.keys():
                
            multplicity = int( rep_dict[ k2 ] )
            
            ### if non-zero, contains one copy of each SO(2) irr
            if ( multplicity != 0 ):
                
                tp = compute_tensor_SO2( k1 , k2 )
               
                for out in tp.keys():
                    
                    try:
                        rep_out[out] = rep_out[out] + int(multplicity)*int(tp[out])
          
                    except:
                        
                        rep_out[out] = int(multplicity)*int(tp[out])
    
    return rep_out


def compute_tensor_SO2_dimension( rep_dict_input  ):
    dval = 0
    for k in rep_dict_input.keys():
        rho_number = rep_dict_input[k]

        if(k==0):
            dval = dval + rho_number
        if (k!=0):
            dval = dval + 2*rho_number

    return dval




