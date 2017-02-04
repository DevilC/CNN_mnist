import theano
import theano.tensor as T

db = T.tensor3('charsdb reshape', 'float64')
reshape_db = db.dimshuffle(2,0,1)
array_reshape = theano.function([db], reshape_db)

reshape_ip = db.dimshuffle(0, 'x', 1, 2)
input_reshape = theano.function([db], reshape_ip)