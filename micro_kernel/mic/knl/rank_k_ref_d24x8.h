  if ( aux->pc ) {
	for ( j = 0; j < 8; j ++ ) {
	  for ( i = 0; i < 24; i ++ ) {
		K[ j * 24 + i ] += c[ j * 24 + i ];
	  }
	}
  }
  for ( p = 0; p < k; p ++ ) {
	for ( j = 0; j < 8; j ++ ) {
	  for ( i = 0; i < 24; i ++ ) {
		K[ j * 24 + i ] += a[ i ] * b [ j ];
	  }
	}
	a += 24;
	b += 8;
  }
