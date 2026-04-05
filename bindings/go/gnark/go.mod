module github.com/carni-ships/zkMetal/bindings/go/gnark

go 1.21

require (
	github.com/carni-ships/zkMetal/bindings/go/zkmetal v0.0.0
	github.com/consensys/gnark-crypto v0.14.0
)

replace github.com/carni-ships/zkMetal/bindings/go/zkmetal => ../zkmetal
