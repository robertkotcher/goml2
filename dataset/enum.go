package dataset

import "github.com/sirupsen/logrus"

// EnumMapper maps "type" -> [instance0, instance1, instance2, etc]
// it's used to easily map back and forth from string to float
type EnumMapper map[string][]string

// Insert adds a new enum type/instance pair (is idempotent)
func (e *EnumMapper) Insert(typ, instance string) {
	found := false
	for _, val := range (*e)[typ] {
		if val == instance {
			found = true
		}
	}
	if !found {
		(*e)[typ] = append((*e)[typ], instance)
	}
}

func (e *EnumMapper) LookupNumFromName(typ, instance string) float64 {
	for i := 0; i < len((*e)[typ]); i++ {
		if (*e)[typ][i] == instance {
			return float64(i)
		}
	}
	logrus.Fatal("could not look up %s.%s in enum mapper", typ, instance)
	return -1
}
