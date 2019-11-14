lpmayos NOTE
-------------

I remove one sentence in en-ud-test.conllu because it contains ids with dots (i.e. 24.1) and makes evaluation fail.
Sure there are more elegant ways to deal with this, but I don't have the time right now.

I remove lines 9394 to 9423:

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
