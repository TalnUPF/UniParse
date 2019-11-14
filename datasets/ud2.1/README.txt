lpmayos NOTE
-------------

I remove one sentence in en-ud-test.conllu and one in en-ud-dev.conllu because they contains ids with dots (i.e. 24.1) and makes evaluation fail.
Sure there are more elegant ways to deal with this, but I don't have the time right now.

dev: I remove lines 19803 to 19832

    # sent_id = answers-20111108072305AAPJTjj_ans-0005
    # text = It's more compact, ISO 6400 capability (SX40 only 3200), faster lens at f/2 and the SX40 only f/2.7.
    1	It	it	PRON	PRP	Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs	4	nsubj	4:nsubj	SpaceAfter=No
    2	's	be	AUX	VBZ	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	4	cop	4:cop	_
    3	more	more	ADV	RBR	_	4	advmod	4:advmod	_
    4	compact	compact	ADJ	JJ	Degree=Pos	0	root	0:root	SpaceAfter=No
    5	,	,	PUNCT	,	_	8	punct	8:punct	_
    6	ISO	iso	NOUN	NN	Number=Sing	8	compound	8:compound	_
    7	6400	6400	NUM	CD	NumType=Card	6	nummod	6:nummod	_
    8	capability	capability	NOUN	NN	Number=Sing	4	list	4:list	_
    9	(	(	PUNCT	-LRB-	_	10	punct	10.1:punct	SpaceAfter=No
    10	SX40	SX40	PROPN	NNP	Number=Sing	8	parataxis	10.1:nsubj	_
    10.1	has	have	VERB	VBZ	_	_	_	8:parataxis	CopyOf=-1
    11	only	only	ADV	RB	_	12	advmod	12:advmod	_
    12	3200	3200	NUM	CD	NumType=Card	10	orphan	10.1:obj	SpaceAfter=No
    13	)	)	PUNCT	-RRB-	_	10	punct	10.1:punct	SpaceAfter=No
    14	,	,	PUNCT	,	_	8	punct	8:punct	_
    15	faster	faster	ADJ	JJR	Degree=Cmp	16	amod	16:amod	_
    16	lens	lens	NOUN	NN	Number=Sing	4	list	4:list	_
    17	at	at	ADP	IN	_	18	case	18:case	_
    18	f/2	f/2	NOUN	NN	Number=Sing	16	nmod	16:nmod	_
    19	and	and	CCONJ	CC	_	21	cc	21.1:cc	_
    20	the	the	DET	DT	Definite=Def|PronType=Art	21	det	21:det	_
    21	SX40	SX40	PROPN	NNP	Number=Sing	16	conj	21.1:nsubj	_
    21.1	has	have	VERB	VBZ	_	_	_	16:conj	CopyOf=-1
    22	only	only	ADJ	JJ	Degree=Pos	23	amod	23:amod	_
    23	f	f	NOUN	NN	Number=Sing	21	orphan	21.1:obj	SpaceAfter=No
    24	/	/	PUNCT	,	_	23	punct	23:punct	SpaceAfter=No
    25	2.7	2.7	NUM	CD	NumType=Card	23	nummod	23:nummod	SpaceAfter=No
    26	.	.	PUNCT	.	_	4	punct	4:punct	_

test: I remove lines 9394 to 9423:

    # sent_id = email-enronsent28_01-0019
    # text = By late 1974 investors were dizzy, they were desperate, they were wrung-out, they had left Wall Street, many for good.
    1	By	by	ADP	IN	_	3	case	3:case	_
    2	late	late	ADJ	JJ	Degree=Pos	3	amod	3:amod	_
    3	1974	1974	NUM	CD	NumType=Card	6	obl	6:obl	_
    4	investors	investor	NOUN	NNS	Number=Plur	6	nsubj	6:nsubj	_
    5	were	be	AUX	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	6	cop	6:cop	_
    6	dizzy	dizzy	ADJ	JJ	Degree=Pos	0	root	0:root	SpaceAfter=No
    7	,	,	PUNCT	,	_	10	punct	10:punct	_|CheckAttachment=6
    8	they	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	10	nsubj	10:nsubj	_
    9	were	be	AUX	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	10	cop	10:cop	_
    10	desperate	desperate	ADJ	JJ	Degree=Pos	6	conj	6:conj	SpaceAfter=No|CheckReln=parataxis
    11	,	,	PUNCT	,	_	14	punct	14:punct	_|CheckAttachment=10
    12	they	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	14	nsubj:pass	14:nsubj:pass	_
    13	were	be	AUX	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	14	aux:pass	14:aux:pass	_
    14	wrung	wring	VERB	VBN	Tense=Past|VerbForm=Part|Voice=Pass	6	conj	6:conj	SpaceAfter=No|CheckReln=parataxis
    15	-	-	PUNCT	HYPH	_	14	punct	14:punct	SpaceAfter=No
    16	out	out	ADP	RP	_	14	compound:prt	14:compound:prt	SpaceAfter=No
    17	,	,	PUNCT	,	_	20	punct	20:punct	_|CheckAttachment=6
    18	they	they	PRON	PRP	Case=Nom|Number=Plur|Person=3|PronType=Prs	20	nsubj	20:nsubj	_
    19	had	have	AUX	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	20	aux	20:aux	_
    20	left	leave	VERB	VBN	Tense=Past|VerbForm=Part	6	conj	6:conj	_|CheckReln=parataxis
    21	Wall	Wall	PROPN	NNP	Number=Sing	22	compound	22:compound	_
    22	Street	Street	PROPN	NNP	Number=Sing	20	obj	20:obj	SpaceAfter=No
    23	,	,	PUNCT	,	_	20	punct	20:punct	_|CheckAttachment=22
    24	many	many	ADJ	JJ	Degree=Pos	6	parataxis	24.1:nsubj	_|CheckAttachment=22|CheckReln=appos
    24.1	left	left	VERB	VBN	Tense=Past|VerbForm=Part	_	_	6:parataxis	CopyOf=6
    25	for	for	ADP	IN	_	26	case	26:case	_
    26	good	good	ADJ	JJ	Degree=Pos	24	orphan	24.1:obl	SpaceAfter=No|CheckReln=nmod
    27	.	.	PUNCT	.	_	6	punct	6:punct	_
