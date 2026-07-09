package ast

type Node interface {
	Pos() (line, col int)
}

type Program struct {
	Line, Col int
	Items     []Item
}

func (n *Program) Pos() (int, int) { return n.Line, n.Col }

type Item interface {
	Node
	item()
}

type FnDef struct {
	Line, Col int
	Name      string
	Params    []Param
	Ret       *TypeAnn
	Body      *Block
}

func (n *FnDef) Pos() (int, int) { return n.Line, n.Col }
func (*FnDef) item()             {}

type ExternDecl struct {
	Line, Col int
	Name      string
	Params    []Param
	Ret       *TypeAnn
}

func (n *ExternDecl) Pos() (int, int) { return n.Line, n.Col }
func (*ExternDecl) item()             {}

type Param struct {
	Line, Col int
	Name      string
	Typ       *TypeAnn
}

func (n *Param) Pos() (int, int) { return n.Line, n.Col }

type TypeAnn struct {
	Line, Col int
	Name      string
	Inner     *TypeAnn
	Params    []*TypeAnn
	Ret       *TypeAnn
}

func (n *TypeAnn) Pos() (int, int) { return n.Line, n.Col }

type Block struct {
	Line, Col int
	Stmts     []Stmt
	Expr      Expr
}

func (n *Block) Pos() (int, int) { return n.Line, n.Col }
func (*Block) expr()             {} // blocks may be expression values

type Stmt interface {
	Node
	stmt()
}

type LetStmt struct {
	Line, Col int
	Name      string
	// Names holds multi-binding destructure targets for let (a, b) = pair; when
	// non-nil/non-empty, Name is the first name for legacy single-name consumers.
	Names []string
	Typ   *TypeAnn
	Value Expr
}

func (n *LetStmt) Pos() (int, int) { return n.Line, n.Col }
func (*LetStmt) stmt()             {}
func (*LetStmt) item()             {}

// BindingNames returns the let binding names (pair-destructure aware, v0.47).
func (n *LetStmt) BindingNames() []string {
	if len(n.Names) > 0 {
		return n.Names
	}
	if n.Name != "" {
		return []string{n.Name}
	}
	return nil
}

type ExprStmt struct {
	Line, Col int
	Expr      Expr
}

func (n *ExprStmt) Pos() (int, int) { return n.Line, n.Col }
func (*ExprStmt) stmt()             {}

type ReturnStmt struct {
	Line, Col int
	Value     Expr
}

func (n *ReturnStmt) Pos() (int, int) { return n.Line, n.Col }
func (*ReturnStmt) stmt()             {}

type Expr interface {
	Node
	expr()
}

type Ternary struct {
	Line, Col        int
	Gate, Pure, Each Expr
}

func (n *Ternary) Pos() (int, int) { return n.Line, n.Col }
func (*Ternary) expr()             {}

type Binary struct {
	Line, Col   int
	Op          string
	Left, Right Expr
}

func (n *Binary) Pos() (int, int) { return n.Line, n.Col }
func (*Binary) expr()             {}

type Unary struct {
	Line, Col int
	Op        string
	Expr      Expr
}

func (n *Unary) Pos() (int, int) { return n.Line, n.Col }
func (*Unary) expr()             {}

type Call struct {
	Line, Col int
	Callee    Expr
	Args      []Expr
}

func (n *Call) Pos() (int, int) { return n.Line, n.Col }
func (*Call) expr()             {}

type Index struct {
	Line, Col int
	Target    Expr
	Index     Expr
}

func (n *Index) Pos() (int, int) { return n.Line, n.Col }
func (*Index) expr()             {}

type Var struct {
	Line, Col int
	Name      string
}

func (n *Var) Pos() (int, int) { return n.Line, n.Col }
func (*Var) expr()             {}

type Literal struct {
	Line, Col int
	Kind      string
	Value     any
}

func (n *Literal) Pos() (int, int) { return n.Line, n.Col }
func (*Literal) expr()             {}

type Lambda struct {
	Line, Col int
	Params    []Param
	Body      Expr
}

func (n *Lambda) Pos() (int, int) { return n.Line, n.Col }
func (*Lambda) expr()             {}

type SpanLit struct {
	Line, Col int
	Elems     []Expr
}

func (n *SpanLit) Pos() (int, int) { return n.Line, n.Col }
func (*SpanLit) expr()             {}
