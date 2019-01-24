
#ifndef RECONSTRUCTFLASHTRANSFORMS_H
#define RECONSTRUCTFLASHTRANSFORMS_H

namespace ants
{
extern int reconstructFlashTransforms( std::vector<std::string>, // equivalent to argv of command line parameters to main()
                                       std::ostream* out_stream  // [optional] output stream to write
                                     );
} // namespace ants

#endif // RECONSTRUCTFLASHTRANSFORMS_H
